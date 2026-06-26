"""Download image bytes for every entry in a NASA manifest.jsonl.

Writes resumable parquet shards; each shard is ~5k rows so memory stays
bounded and a timeout doesn't lose the whole job.

License classification is applied inline (nasa_id, center, secondary_creator,
description are checked for ESA / third-party credit signals). Every record
gets a `license` column: "PD" (NASA-owned), "CC-BY-SA-3.0-IGO" (ESA-joint),
or "UNCLEAR" (third-party credits — drop before training).

Usage:
    python download.py --manifest ... --out-dir ... --concurrency 32 --rps 30
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

# Signals that an image has non-pure-NASA provenance.
_ESA_RE = re.compile(r"\b(ESA|European Space Agency|Hubble Heritage Team)\b", re.I)
# Explicit third-party credit patterns in the description field.
_THIRD_PARTY_RE = re.compile(
    r"\b(courtesy of|photo by|copyright|©|getty images|ap photo|reuters)\b", re.I
)


def classify_license(rec: dict) -> str:
    hay = " ".join([
        rec.get("center", ""),
        rec.get("secondary_creator", ""),
        rec.get("description", ""),
    ])
    if _THIRD_PARTY_RE.search(hay):
        return "UNCLEAR"
    if _ESA_RE.search(hay):
        return "CC-BY-SA-3.0-IGO"
    return "PD"


class TokenBucket:
    """Async token bucket for client-side rate limiting."""
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


async def fetch_one(client: httpx.AsyncClient, bucket: TokenBucket, rec: dict,
                    metrics: dict) -> dict | None:
    url = rec["image_url"]
    for attempt in range(4):
        try:
            await bucket.acquire()
            r = await client.get(url, headers=HEADERS, timeout=120.0,
                                 follow_redirects=True)
            if r.status_code == 404:
                metrics["404"] += 1
                return None
            if r.status_code == 429:
                await asyncio.sleep(min(30, 5 * (2 ** attempt)))
                continue
            r.raise_for_status()
            out = dict(rec)
            out["image_bytes"] = r.content
            out["license"] = classify_license(rec)
            out["keywords_json"] = json.dumps(rec.get("keywords") or [],
                                               ensure_ascii=False)
            out.pop("keywords", None)
            metrics["ok"] += 1
            metrics[f"license_{out['license']}"] = \
                metrics.get(f"license_{out['license']}", 0) + 1
            return out
        except Exception:
            if attempt == 3:
                metrics["err"] += 1
                return None
            await asyncio.sleep(2 ** attempt)
    return None


def shard_path(out_dir: Path, idx: int) -> Path:
    return out_dir / f"nasa_images_{idx:05d}.parquet"


def load_done_ids(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for p in sorted(out_dir.glob("nasa_images_*.parquet")):
        try:
            ids = pl.read_parquet(p, columns=["nasa_id"])["nasa_id"].to_list()
        except Exception as e:
            # Shard truncated by SIGKILL/OOM — delete so resume rebuilds it.
            print(f"  removing corrupt shard {p.name}: {str(e)[:80]}", flush=True)
            p.unlink()
            continue
        done.update(ids)
    return done


async def main_async(args):
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = []
    with open(args.manifest) as f:
        for line in f:
            recs.append(json.loads(line))

    done = load_done_ids(out_dir) if not args.overwrite else set()
    print(f"manifest: {len(recs):,} records", flush=True)
    print(f"already downloaded: {len(done):,}", flush=True)

    todo = [r for r in recs if r["nasa_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"to fetch: {len(todo):,}", flush=True)
    if not todo:
        return

    bucket = TokenBucket(args.rps)
    sem = asyncio.Semaphore(args.concurrency)
    metrics = {"ok": 0, "err": 0, "404": 0}

    # Shard index continues from existing shards so we don't overwrite.
    existing = list(out_dir.glob("nasa_images_*.parquet"))
    shard_idx = len(existing)

    async def bounded(client, rec):
        async with sem:
            return await fetch_one(client, bucket, rec, metrics)

    buf: list[dict] = []
    buf_bytes = 0                         # running byte total for size-aware flush
    t0 = time.time()
    flush_every_rows = args.flush_every
    flush_every_bytes = args.flush_bytes  # backstop when images are unexpectedly large

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrency * 2),
    ) as client:
        tasks = [asyncio.create_task(bounded(client, r)) for r in todo]
        for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
            row = await coro
            if row is not None:
                buf.append(row)
                buf_bytes += len(row.get("image_bytes", b""))
            if i % 200 == 0:
                dt = time.time() - t0
                print(f"[{i:>6}/{len(todo)}] ok={metrics['ok']} "
                      f"err={metrics['err']} 404={metrics['404']}  "
                      f"rate={i/max(dt,1):.1f}/s  "
                      f"buf={len(buf)} rows/{buf_bytes/1e9:.1f}GB", flush=True)
            if len(buf) >= flush_every_rows or buf_bytes >= flush_every_bytes:
                p = shard_path(out_dir, shard_idx)
                pl.DataFrame(buf).write_parquet(p, compression="zstd")
                size_mb = p.stat().st_size / 1e6
                print(f"    shard {shard_idx}: {p.name} "
                      f"({len(buf):,} rows, {size_mb:.0f} MB)", flush=True)
                shard_idx += 1
                buf.clear()
                buf_bytes = 0

    if buf:
        p = shard_path(out_dir, shard_idx)
        pl.DataFrame(buf).write_parquet(p, compression="zstd")
        size_mb = p.stat().st_size / 1e6
        print(f"    shard {shard_idx}: {p.name} "
              f"({len(buf):,} rows, {size_mb:.0f} MB)", flush=True)
        buf.clear()
        buf_bytes = 0

    dt = time.time() - t0
    print(f"\nDONE in {dt/60:.1f} min. Final:", flush=True)
    for k, v in sorted(metrics.items()):
        print(f"  {k:<30} {v:,}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--rps", type=float, default=30.0)
    ap.add_argument("--flush-every", type=int, default=500,
                    help="flush shard after this many rows")
    ap.add_argument("--flush-bytes", type=int, default=8 * 1024**3,
                    help="flush shard after buffered image bytes exceed this (default 8GB)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
