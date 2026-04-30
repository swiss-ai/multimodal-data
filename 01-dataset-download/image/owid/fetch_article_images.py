"""Download image bytes for every image segment in articles_interleave.parquet.

Input:  articles_interleave.parquet  (segments already parsed — type, url, alt)
Output: article_images.parquet       (slug, kind, segment_idx, url, alt,
                                       img_kind, image_bytes, content_type)

Per-URL bytes are fetched once even if the same image recurs across multiple
articles, so we don't hammer the CDN for duplicates.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import polars as pl


HEADERS = {"User-Agent": "Mozilla/5.0"}


class TokenBucket:
    def __init__(self, rate_per_sec: float):
        self.rate = rate_per_sec
        self.tokens = rate_per_sec
        self.updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens < 1:
                await asyncio.sleep((1 - self.tokens) / self.rate)
                self.tokens = 0
            else:
                self.tokens -= 1


async def fetch_one(client: httpx.AsyncClient, bucket: TokenBucket, url: str,
                    metrics: dict) -> tuple[bytes | None, str]:
    for attempt in range(4):
        try:
            await bucket.acquire()
            r = await client.get(url, headers=HEADERS, timeout=60.0,
                                 follow_redirects=True)
            if r.status_code == 404:
                metrics["404"] += 1
                return None, ""
            if r.status_code == 429:
                await asyncio.sleep(min(30, 5 * (2 ** attempt)))
                continue
            r.raise_for_status()
            metrics["ok"] += 1
            return r.content, r.headers.get("content-type", "")
        except Exception:
            if attempt == 3:
                metrics["err"] += 1
                return None, ""
            await asyncio.sleep(2 ** attempt)
    return None, ""


def enumerate_image_segments(interleave_path: Path) -> list[dict]:
    df = pl.read_parquet(interleave_path)
    out = []
    for row in df.iter_rows(named=True):
        segs = json.loads(row["segments_json"])
        for idx, s in enumerate(segs):
            if not isinstance(s, dict):
                continue
            if s.get("type") != "image":
                continue
            url = s.get("url")
            if not url:
                continue
            out.append({
                "slug": row["slug"],
                "kind": row["kind"],
                "segment_idx": idx,
                "url": url,
                "alt": s.get("alt") or "",
                "img_kind": s.get("img_kind") or "",
            })
    return out


async def main_async(args):
    seg_rows = enumerate_image_segments(args.interleave)
    print(f"image segments total: {len(seg_rows):,}", flush=True)
    urls = sorted({r["url"] for r in seg_rows})
    print(f"unique URLs:          {len(urls):,}", flush=True)

    # Resume: if the output exists, skip URLs we've already fetched.
    out_path = args.out
    already: set[str] = set()
    if out_path.exists() and not args.overwrite:
        prev = pl.read_parquet(out_path, columns=["url"])
        already = set(prev["url"].to_list())
        print(f"resume: {len(already):,} URLs already in output", flush=True)

    url_todo = [u for u in urls if u not in already]
    if args.limit:
        url_todo = url_todo[: args.limit]
    print(f"URLs to fetch: {len(url_todo):,}", flush=True)
    if not url_todo and not args.overwrite:
        # Still may need to emit segment rows if we'd previously only written
        # url-keyed rows. Skipping — caller can re-run with --overwrite.
        print("nothing new to fetch", flush=True)
        return

    bucket = TokenBucket(args.rps)
    sem = asyncio.Semaphore(args.concurrency)
    metrics = {"ok": 0, "err": 0, "404": 0}
    url_to_bytes: dict[str, tuple[bytes, str]] = {}

    async def bounded(client, url):
        async with sem:
            b, ct = await fetch_one(client, bucket, url, metrics)
            if b is not None:
                url_to_bytes[url] = (b, ct)

    t0 = time.time()
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrency * 2),
    ) as client:
        tasks = [asyncio.create_task(bounded(client, u)) for u in url_todo]
        for i, _ in enumerate(asyncio.as_completed(tasks), start=1):
            await _
            if i % 200 == 0:
                dt = time.time() - t0
                print(f"[{i:>6}/{len(url_todo)}] ok={metrics['ok']} "
                      f"err={metrics['err']} 404={metrics['404']}  "
                      f"rate={i/max(dt,1):.1f}/s", flush=True)

    dt = time.time() - t0
    print(f"\nfetch done in {dt/60:.1f} min: {metrics}", flush=True)

    # Emit one row per image segment (bytes shared across duplicates).
    print(f"assembling output rows from {len(seg_rows):,} segments...", flush=True)
    out_rows = []
    for r in seg_rows:
        cached = url_to_bytes.get(r["url"])
        if cached is None and r["url"] in already:
            # Already in the existing parquet; we'll union below.
            continue
        if cached is None:
            continue
        b, ct = cached
        out_rows.append({**r, "image_bytes": b, "content_type": ct})

    if not out_rows and not already:
        print("no rows to write", flush=True)
        return

    new_df = pl.DataFrame(out_rows) if out_rows else pl.DataFrame()
    if out_path.exists() and not args.overwrite:
        existing = pl.read_parquet(out_path)
        combined = pl.concat([existing, new_df], how="diagonal_relaxed") if len(new_df) else existing
    else:
        combined = new_df

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path, compression="zstd")
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  ({len(combined):,} rows, {size_mb:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interleave", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/owid___charts/articles_interleave.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/owid___charts/article_images.parquet"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rps", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only fetch first N unique URLs (smoke test).")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
