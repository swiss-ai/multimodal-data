"""Enumerate the NASA Image and Video Library into a flat JSONL manifest.

Uses the public images-api.nasa.gov search endpoint, paginated by year so
no single query exceeds the API's 10,000-result cap. Writes one JSON line
per image with metadata + the URL of the highest-resolution ``~orig.jpg``.

Usage:
    python enumerate.py --out /capstor/.../web___nasa___images/manifest.jsonl
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx


BASE = "https://images-api.nasa.gov/search"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def extract_image_url(links: list[dict]) -> tuple[str | None, int | None, int | None]:
    """Pick the highest-resolution image URL from the API links array.

    Order preference: rel=canonical (orig), else rel=alternate sorted by
    width desc. Returns (url, width, height) or (None, None, None).
    """
    canon = next((l for l in links if l.get("rel") == "canonical"
                  and l.get("render") == "image"), None)
    if canon:
        return canon.get("href"), canon.get("width"), canon.get("height")
    alts = [l for l in links if l.get("rel") == "alternate"
            and l.get("render") == "image" and l.get("href")]
    if not alts:
        return None, None, None
    alts.sort(key=lambda l: l.get("width") or 0, reverse=True)
    top = alts[0]
    return top.get("href"), top.get("width"), top.get("height")


async def fetch_page(client: httpx.AsyncClient, year_s: int, year_e: int,
                     page: int, metrics: dict) -> list[dict]:
    params = {
        "media_type": "image",
        "year_start": year_s,
        "year_end": year_e,
        "page": page,
    }
    for attempt in range(4):
        try:
            r = await client.get(BASE, headers=HEADERS, params=params, timeout=60.0)
            if r.status_code == 429:
                await asyncio.sleep(5 * (2 ** attempt))
                continue
            r.raise_for_status()
            coll = r.json().get("collection", {})
            return coll.get("items", [])
        except Exception as e:
            if attempt == 3:
                metrics["page_err"] += 1
                print(f"  FAIL page {year_s}-{year_e}/{page}: "
                      f"{type(e).__name__}: {str(e)[:150]}", flush=True)
                return []
            await asyncio.sleep(2 ** attempt)
    return []


def normalize_item(item: dict) -> dict | None:
    data = (item.get("data") or [{}])[0]
    nasa_id = data.get("nasa_id")
    if not nasa_id:
        return None
    img_url, w, h = extract_image_url(item.get("links") or [])
    if not img_url:
        return None
    return {
        "nasa_id": nasa_id,
        "title": data.get("title") or "",
        "description": data.get("description") or "",
        "keywords": data.get("keywords") or [],
        "center": data.get("center") or "",
        "secondary_creator": data.get("secondary_creator") or "",
        "date_created": data.get("date_created") or "",
        "image_url": img_url,
        "image_width": w,
        "image_height": h,
    }


async def main_async(args):
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    if out_path.exists() and not args.overwrite:
        with out_path.open() as f:
            for line in f:
                seen_ids.add(json.loads(line)["nasa_id"])
        print(f"resume: {len(seen_ids):,} ids already in manifest", flush=True)

    metrics = {"pages": 0, "page_err": 0, "new": 0, "dup": 0}
    t0 = time.time()

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrency * 2),
    ) as client:
        # Work plan: year-at-a-time. API caps results per search; pages run
        # until empty, up to max_pages safety.
        fh = out_path.open("a")
        for year in range(args.year_start, args.year_end + 1):
            for page in range(1, args.max_pages + 1):
                items = await fetch_page(client, year, year, page, metrics)
                metrics["pages"] += 1
                if not items:
                    break
                for item in items:
                    rec = normalize_item(item)
                    if rec is None:
                        continue
                    if rec["nasa_id"] in seen_ids:
                        metrics["dup"] += 1
                        continue
                    seen_ids.add(rec["nasa_id"])
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    metrics["new"] += 1
                fh.flush()
                if len(items) < 100:
                    break  # last page for this year
            dt = time.time() - t0
            print(f"  year={year} pages={metrics['pages']} new={metrics['new']:,} "
                  f"dup={metrics['dup']:,} err={metrics['page_err']}  "
                  f"rate={metrics['pages']/max(dt,1):.1f} pages/s", flush=True)
        fh.close()

    dt = time.time() - t0
    print(f"\nDONE in {dt/60:.1f} min. Final: {metrics}", flush=True)
    print(f"  manifest: {out_path}  ({len(seen_ids):,} unique nasa_ids)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--year-start", type=int, default=1958)
    ap.add_argument("--year-end", type=int, default=2026)
    ap.add_argument("--max-pages", type=int, default=100,
                    help="Max pages per year (safety bound; each page is 100 hits).")
    ap.add_argument("--concurrency", type=int, default=4,
                    help="Parallel HTTP connections for the search API.")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
