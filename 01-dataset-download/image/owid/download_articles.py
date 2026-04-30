"""Fetch raw HTML for OWID narrative articles and data-insights.

Output: ONE parquet (articles.parquet) holding raw HTML for each slug.
Per record: kind, slug, url, html, fetched_at, http_status, license.

Downstream processing (extraction of paragraph+embedded-chart pairs) is
decoupled from this raw-acquisition step so we don't re-hit the server if
the extractor evolves.

Usage:
    python download_articles.py \
        --slugs-jsonl raw/cooldown/owid___charts/slugs.jsonl \
        --out-dir    raw/cooldown/owid___charts \
        --kinds      article data_insight
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import polars as pl

from download_graphers import TokenBucket, _append_parquet, HEADERS


KIND_URLS = {
    "article": "https://ourworldindata.org/{slug}",
    "data_insight": "https://ourworldindata.org/data-insights/{slug}",
}


async def fetch_html(client: httpx.AsyncClient, bucket: TokenBucket,
                     kind: str, slug: str, metrics: dict) -> dict | None:
    url = KIND_URLS[kind].format(slug=slug)
    for attempt in range(4):
        try:
            await bucket.acquire()
            r = await client.get(url, headers=HEADERS, timeout=60.0,
                                 follow_redirects=True)
            if r.status_code == 404:
                metrics["404"] += 1
                return None
            if r.status_code == 429:
                await asyncio.sleep(min(30, 2 ** attempt * 5))
                continue
            r.raise_for_status()
            metrics["ok"] += 1
            return {
                "kind": kind,
                "slug": slug,
                "url": url,
                "html": r.text,
                "http_status": r.status_code,
                "fetched_at": int(time.time()),
                "license": "CC-BY-4.0",
            }
        except Exception:
            if attempt == 3:
                metrics["err"] += 1
                return None
            await asyncio.sleep(2 ** attempt)
    return None


async def main_async(args):
    todo: list[tuple[str, str]] = []
    with open(args.slugs_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] in args.kinds:
                todo.append((r["kind"], r["slug"]))

    out_path = args.out_dir / "articles.parquet"
    done_keys: set[tuple[str, str]] = set()
    if out_path.exists() and not args.overwrite:
        done = pl.read_parquet(out_path, columns=["kind", "slug"])
        done_keys = set(zip(done["kind"].to_list(), done["slug"].to_list()))
        print(f"resume: {len(done_keys):,} already downloaded", flush=True)

    remaining = [(k, s) for (k, s) in todo if (k, s) not in done_keys]
    print(f"to fetch: {len(remaining):,} / {len(todo):,} total", flush=True)
    if args.limit:
        remaining = remaining[: args.limit]
        print(f"  limited to first {args.limit}", flush=True)
    if not remaining:
        print("nothing to do", flush=True)
        return

    bucket = TokenBucket(args.rps)
    metrics = {"ok": 0, "err": 0, "404": 0}
    sem = asyncio.Semaphore(args.concurrency)

    async def bounded(client, kind, slug):
        async with sem:
            return await fetch_html(client, bucket, kind, slug, metrics)

    buf: list[dict] = []
    t0 = time.time()
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrency * 2),
    ) as client:
        tasks = [asyncio.create_task(bounded(client, k, s)) for k, s in remaining]
        for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
            row = await coro
            if row is not None:
                buf.append(row)
            if i % 50 == 0:
                dt = time.time() - t0
                print(f"[{i:>5}/{len(remaining)}] ok={metrics['ok']} "
                      f"404={metrics['404']} err={metrics['err']}  "
                      f"rate={i/dt:.1f}/s", flush=True)
            if len(buf) >= args.flush_every:
                _append_parquet(out_path, buf)
                buf.clear()

    if buf:
        _append_parquet(out_path, buf)

    dt = time.time() - t0
    print(f"\nDONE in {dt/60:.1f} min:", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v:,}", flush=True)
    print(f"  output: {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--kinds", nargs="+", default=["article", "data_insight"],
                    choices=list(KIND_URLS))
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--rps", type=float, default=3.0)
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
