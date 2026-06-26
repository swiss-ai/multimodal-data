"""Build ONE canonical data_insights.parquet: one row per insight,
clean main-body text + the main chart image bytes + sidecar metadata.

Workflow:
  1. Start from data_insights_clean.parquet (416 rows, high-quality body text).
  2. For each insight, find the main chart image segment from
     articles_interleave.parquet. Main = first image that isn't the author
     portrait. Portraits are small-width (w<=600) with alt = author name.
  3. Fetch each unique main-image URL from the CDN (parallel, rate-limited).
  4. Emit data_insights.parquet with schema:
       slug, title, body_text, word_count, authors_json, url, license,
       image_bytes, image_url, image_alt, chart_slugs_json

Sidecar metadata (author, url, license, chart slugs) lives inside the same
parquet; downstream consumers can ignore them if they want image+text only.
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import polars as pl


HEADERS = {"User-Agent": "Mozilla/5.0"}
_WIDTH_RE = re.compile(r"/w=(\d+)$")


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


def pick_main_image(segments: list, author_names: set[str]) -> dict | None:
    """First image block that isn't an author portrait.

    Author portraits have ``alt == author_name`` verbatim (regardless of the
    width variant the CDN returns). Main chart images have descriptive alt
    (e.g. "Line chart showing...").
    """
    for s in segments:
        if not isinstance(s, dict) or s.get("type") != "image":
            continue
        alt = (s.get("alt") or "").strip()
        if alt in author_names:
            continue
        return s
    return None


async def fetch_one(client: httpx.AsyncClient, bucket: TokenBucket,
                    url: str, metrics: dict) -> tuple[bytes | None, str]:
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


async def main_async(args):
    clean = pl.read_parquet(args.clean).filter(pl.col("kind") == "data_insight")
    interleave = pl.read_parquet(args.interleave).filter(
        pl.col("kind") == "data_insight")
    # Build slug -> (title, body_text, word_count, authors_json, chart_slugs_json,
    #                url, license) from clean
    clean_by_slug = {r["slug"]: r for r in clean.iter_rows(named=True)}
    # slug -> segments
    segs_by_slug = {r["slug"]: json.loads(r["segments_json"])
                    for r in interleave.iter_rows(named=True)}
    print(f"data_insights in clean:      {len(clean_by_slug):,}", flush=True)
    print(f"data_insights in interleave: {len(segs_by_slug):,}", flush=True)

    slugs = sorted(set(clean_by_slug) & set(segs_by_slug))
    print(f"slugs covered in both:       {len(slugs):,}", flush=True)

    # For each slug, resolve the main image URL
    rows_plan: list[dict] = []
    n_no_image = 0
    for slug in slugs:
        c = clean_by_slug[slug]
        authors = json.loads(c.get("authors_json") or "[]")
        author_names = {a for a in authors if a}
        main = pick_main_image(segs_by_slug[slug], author_names)
        if main is None:
            n_no_image += 1
            continue
        rows_plan.append({
            "slug": slug,
            "title": c.get("title") or "",
            "body_text": c.get("body_text") or "",
            "word_count": c.get("word_count") or 0,
            "authors_json": c.get("authors_json") or "[]",
            "url": c.get("url") or "",
            "license": c.get("license") or "CC-BY-4.0",
            "image_url": main["url"],
            "image_alt": main.get("alt") or "",
            "chart_slugs_json": c.get("chart_slugs_json") or "[]",
        })
    print(f"planned rows: {len(rows_plan):,}  (skipped {n_no_image} with no suitable image)", flush=True)
    if not rows_plan:
        return

    # Fetch unique URLs
    url_set = sorted({r["image_url"] for r in rows_plan})
    print(f"unique image URLs to fetch: {len(url_set):,}", flush=True)

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
        tasks = [asyncio.create_task(bounded(client, u)) for u in url_set]
        for i, _ in enumerate(asyncio.as_completed(tasks), start=1):
            await _
            if i % 100 == 0:
                dt = time.time() - t0
                print(f"[{i:>5}/{len(url_set)}] ok={metrics['ok']} "
                      f"err={metrics['err']} 404={metrics['404']}  "
                      f"rate={i/max(dt,1):.1f}/s", flush=True)

    print(f"\nfetch done in {(time.time()-t0)/60:.1f} min: {metrics}", flush=True)

    # Assemble final rows with image bytes
    final_rows = []
    for r in rows_plan:
        cached = url_to_bytes.get(r["image_url"])
        if cached is None:
            continue
        b, ct = cached
        final_rows.append({**r, "image_bytes": b, "content_type": ct})

    print(f"rows with image_bytes: {len(final_rows):,}", flush=True)
    if not final_rows:
        return

    out = pl.DataFrame(final_rows).select([
        "slug", "title", "body_text", "word_count", "authors_json",
        "url", "license", "image_bytes", "image_url", "image_alt",
        "content_type", "chart_slugs_json",
    ])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out, compression="zstd")
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({len(out):,} rows, {size_mb:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts/data_insights_clean.parquet"))
    ap.add_argument("--interleave", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/owid___charts/articles_interleave.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts/data_insights.parquet"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rps", type=float, default=15.0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
