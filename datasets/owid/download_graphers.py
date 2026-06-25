"""Fetch {slug}.png + {slug}.config.json for every OWID grapher slug.

Filter kept for training use:
  - HTTP 200 on both endpoints
  - config.json has non-empty "dimensions" (i.e. it's an OWID-rendered chart,
    not a third-party embed)

Rate-limited at --rps (default 5 req/s aggregate) via a simple token bucket so
we don't get Cloudflare-slapped. Resumable: skips slugs already present in the
output parquet.

Writes ONE parquet:
    grapher_charts.parquet  (slug, image_bytes, title, subtitle, sources_line,
                             source_descs_json, config_json, license,
                             attribution_page)

Usage:
    python download_graphers.py \
        --slugs-jsonl raw/cooldown/owid___charts/slugs.jsonl \
        --out-dir    raw/cooldown/owid___charts \
        --concurrency 8 --rps 5
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import polars as pl


OWID_BASE = "https://ourworldindata.org"
PNG_URL = OWID_BASE + "/grapher/{slug}.png?width=1200"
CFG_URL = OWID_BASE + "/grapher/{slug}.config.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}


class TokenBucket:
    """Simple async token bucket for rate limiting."""
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
                wait = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0
            else:
                self.tokens -= 1


async def fetch_one(client: httpx.AsyncClient, bucket: TokenBucket, slug: str,
                    metrics: dict) -> dict | None:
    png_url = PNG_URL.format(slug=slug)
    cfg_url = CFG_URL.format(slug=slug)

    # config first — cheap, and we can skip the PNG if it's not OWID-rendered.
    for attempt in range(4):
        try:
            await bucket.acquire()
            r = await client.get(cfg_url, headers=HEADERS, timeout=30.0)
            if r.status_code == 404:
                metrics["cfg_404"] += 1
                return None
            if r.status_code == 429:
                await asyncio.sleep(min(30, 2 ** attempt * 5))
                continue
            r.raise_for_status()
            cfg = r.json()
            break
        except Exception as e:
            if attempt == 3:
                metrics["cfg_err"] += 1
                return None
            await asyncio.sleep(2 ** attempt)
    else:
        return None

    # OWID-rendered filter: needs non-empty dimensions.
    if not cfg.get("dimensions"):
        metrics["not_rendered"] += 1
        return None
    if not cfg.get("isPublished", True):
        metrics["unpublished"] += 1
        return None

    # PNG
    for attempt in range(4):
        try:
            await bucket.acquire()
            r = await client.get(png_url, headers=HEADERS, timeout=60.0)
            if r.status_code == 404:
                metrics["png_404"] += 1
                return None
            if r.status_code == 429:
                await asyncio.sleep(min(30, 2 ** attempt * 5))
                continue
            r.raise_for_status()
            png_bytes = r.content
            break
        except Exception:
            if attempt == 3:
                metrics["png_err"] += 1
                return None
            await asyncio.sleep(2 ** attempt)
    else:
        return None

    metrics["ok"] += 1

    # Pull the per-variable source descriptions into a single JSON blob.
    src_descs = []
    for dim in cfg.get("dimensions", []):
        disp = dim.get("display") or {}
        src_descs.append({
            "variableId": dim.get("variableId"),
            "property": dim.get("property"),
            "display_name": disp.get("name"),
            "unit": disp.get("unit"),
        })

    return {
        "slug": slug,
        "image_bytes": png_bytes,
        "title": cfg.get("title") or "",
        "subtitle": cfg.get("subtitle") or "",
        "source_line": cfg.get("sourceDesc") or "",
        "note": cfg.get("note") or "",
        "source_descs_json": json.dumps(src_descs, ensure_ascii=False),
        "config_json": json.dumps(cfg, ensure_ascii=False),
        "license": "CC-BY-4.0",
        "attribution_url": f"{OWID_BASE}/grapher/{slug}",
    }


async def main_async(args):
    slugs = []
    with open(args.slugs_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "grapher":
                slugs.append(r["slug"])

    out_path = args.out_dir / "grapher_charts.parquet"
    done_slugs: set[str] = set()
    if out_path.exists() and not args.overwrite:
        done = pl.read_parquet(out_path, columns=["slug"])
        done_slugs = set(done["slug"].to_list())
        print(f"resume: {len(done_slugs):,} already downloaded", flush=True)

    remaining = [s for s in slugs if s not in done_slugs]
    print(f"to fetch: {len(remaining):,} / {len(slugs):,} total", flush=True)
    if args.limit:
        remaining = remaining[: args.limit]
        print(f"  limited to first {args.limit}", flush=True)
    if not remaining:
        print("nothing to do", flush=True)
        return

    bucket = TokenBucket(args.rps)
    metrics = {"ok": 0, "cfg_404": 0, "cfg_err": 0, "png_404": 0,
               "png_err": 0, "not_rendered": 0, "unpublished": 0}

    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(client, slug):
        async with sem:
            return await fetch_one(client, bucket, slug, metrics)

    t0 = time.time()
    flush_every = args.flush_every
    buf: list[dict] = []

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrency * 2),
    ) as client:
        tasks = [asyncio.create_task(bounded(client, s)) for s in remaining]
        for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
            row = await coro
            if row is not None:
                buf.append(row)
            if i % 50 == 0:
                dt = time.time() - t0
                print(f"[{i:>5}/{len(remaining)}] ok={metrics['ok']} "
                      f"not_rendered={metrics['not_rendered']} "
                      f"cfg_404={metrics['cfg_404']} png_404={metrics['png_404']} "
                      f"err={metrics['cfg_err'] + metrics['png_err']}  "
                      f"rate={i/dt:.1f}/s", flush=True)
            if len(buf) >= flush_every:
                _append_parquet(out_path, buf)
                buf.clear()

    if buf:
        _append_parquet(out_path, buf)

    dt = time.time() - t0
    print(f"\nDONE in {dt/60:.1f} min:", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v:,}", flush=True)
    print(f"  output: {out_path}", flush=True)


def _append_parquet(path: Path, rows: list[dict]):
    new_df = pl.DataFrame(rows)
    if path.exists():
        old = pl.read_parquet(path)
        combined = pl.concat([old, new_df], how="diagonal_relaxed")
    else:
        combined = new_df
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(path, compression="zstd")
    print(f"    flushed {len(rows)} new rows -> {path} "
          f"(total {len(combined):,})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--rps", type=float, default=5.0,
                    help="Aggregate request rate cap (req/s).")
    ap.add_argument("--flush-every", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only fetch the first N (for smoke tests).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Ignore existing output and start from scratch.")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
