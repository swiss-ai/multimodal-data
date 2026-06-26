"""Build articles.parquet from the OWID GDOC body blocks (not from the
pre-flattened interleave). The GDOC body contains ONLY real article content —
no license boilerplate, no acknowledgments, no endnotes, no newsletter signups,
no topic-page chrome. Those all live outside `content.body` in the page.

Flow:
  1. For each article, parse the GDOC JSON embedded in the raw HTML.
  2. Walk `content.body` blocks in order; convert each to a clean segment:
       type=text/heading/callout  →  rendered text
       type=image                 →  image reference (filename)
       type=horizontal-rule       →  skipped
  3. Resolve each image filename to an actual URL by order-matching against
     the pre-parsed articles_interleave segments (which already has the
     CDN URLs from the rendered page).
  4. Fetch CDN URLs that aren't already in grapher_charts.parquet.
  5. Emit: slug, title, subtitle, url, license, authors_json, body_text (flat),
     word_count, n_segments, n_images, segments_json, images_bytes, images_meta_json.
"""
import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx
import polars as pl

from extract_bodies import _extract_gdoc


HEADERS = {"User-Agent": "Mozilla/5.0"}
GRAPHER_SLUG_RE = re.compile(r"/grapher/([^/?#]+)")

# Footer cut: truncate the article body at the first segment whose rendered
# text starts with any of these markers (with or without leading # heading
# markers). Applied to GDOC-rendered blocks.
_CUT_MARKERS = [
    r"Reuse this work freely", r"All visualizations, data, and code",
    r"Cite this work",
    r"Endnotes?\b", r"References\b", r"Bibliography\b", r"Notes\b",
    r"Acknowledgments?\b", r"Acknowledgements?\b",
    r"I would like to thank\s+[A-Z]",
    r"We thank\s+[A-Z]",
    r"Many thanks to\s+[A-Z]",
    r"Thanks to\s+[A-Z][\w\.\-']+",
    r"Keep reading on Our World in Data",
    r"Explore more research and data on",
]
_CUT_RE = re.compile(r"^(?:#+\s*)?(?:" + "|".join(_CUT_MARKERS) + r")",
                      re.IGNORECASE)

# Per-block drop patterns (individual paragraphs / cards to remove without
# truncating the rest of the article).
_DROP_PATTERNS = [
    r"Subscribe to our newsletters?", r"Sign up for our newsletters?",
    r"We send two regular newsletters", r"^Subscribe\s*$",
    r"Get our newsletter",
    r"^Related chart", r"^Related article",
    r"We have more recent data on this topic",
]
_DROP_RE = re.compile(r"^(?:#+\s*)?(?:" + "|".join(_DROP_PATTERNS) + r")",
                       re.IGNORECASE)

_CITATION_YEAR_RE = re.compile(r"\([12]\d{3}\)")


def _is_cut_text(text: str) -> bool:
    return bool(_CUT_RE.match(re.sub(r"^[#\s]+", "", text)))


def _is_drop_text(text: str) -> bool:
    return bool(_DROP_RE.match(re.sub(r"^[#\s]+", "", text)))


def _is_reference_like(text: str) -> bool:
    t = re.sub(r"^[#\s]+", "", text)
    if not t or len(t.split()) > 300:
        return False
    head = t[:400]
    return bool(_CITATION_YEAR_RE.search(head) and
                re.match(r"^[A-Za-z][\w\-']+\s*,", head))


class TokenBucket:
    def __init__(self, rate: float):
        self.rate = rate; self.tokens = rate; self.updated = time.monotonic()
        self._lock = asyncio.Lock()
    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens < 1:
                await asyncio.sleep((1 - self.tokens) / self.rate); self.tokens = 0
            else:
                self.tokens -= 1


async def fetch(client, bucket, url, metrics):
    for attempt in range(4):
        try:
            await bucket.acquire()
            r = await client.get(url, headers=HEADERS, timeout=60.0, follow_redirects=True)
            if r.status_code == 404:
                metrics["404"] += 1; return None
            if r.status_code == 429:
                await asyncio.sleep(min(30, 5 * (2 ** attempt))); continue
            r.raise_for_status()
            metrics["ok"] += 1
            return r.content
        except Exception:
            if attempt == 3:
                metrics["err"] += 1; return None
            await asyncio.sleep(2 ** attempt)
    return None


# ---------- GDOC rendering ---------------------------------------------------

def _render_spans(spans) -> str:
    if not isinstance(spans, list):
        return ""
    out = []
    for s in spans:
        if not isinstance(s, dict):
            continue
        t = s.get("spanType")
        if t == "span-simple-text":
            out.append(s.get("text", ""))
        elif t == "span-newline":
            out.append("\n")
        else:
            out.append(_render_spans(s.get("children") or []))
    return "".join(out)


def _render_text_block(b: dict) -> str:
    return _render_spans(b.get("value") or [])


def _render_heading_block(b: dict) -> str:
    level = max(1, min(6, int(b.get("level") or 1) + 1))
    text = _render_spans(b.get("text") or [])
    return f"{'#' * level} {text}".strip()


def _render_callout_block(b: dict) -> str:
    """Callouts are titled text boxes — article summary, tip, note. Keep the
    body text; omit the title (it's usually just 'Summary' or 'Tip')."""
    parts = []
    for child in b.get("text") or []:
        if isinstance(child, dict) and child.get("type") == "text":
            parts.append(_render_spans(child.get("value") or []))
    return "\n\n".join(p for p in parts if p.strip())


def _render_list_block(b: dict) -> str:
    items = b.get("items") or []
    rendered = []
    for it in items:
        if isinstance(it, dict):
            rendered.append(_render_spans(it.get("value") or []))
    return "\n".join(f"- {x}" for x in rendered if x)


_HEADING_RE = re.compile(r"^(#{1,6})\s")


def _heading_level(seg: dict) -> int | None:
    """Markdown heading level for a segment, else None."""
    if seg.get("type") != "text":
        return None
    m = _HEADING_RE.match(seg.get("value", ""))
    return len(m.group(1)) if m else None


def reorder_by_section(segs: list[dict]) -> list[dict]:
    """Reorder segments so that within each heading-delimited section (any
    level: H1, H2, H3, …), the heading leads, then all image segments, then
    the section's text segments.

    Splitting on every heading (not just H2) keeps each image with its
    closest-subsection text. This matters for articles that nest many H3
    subsections under one H2 — dumping every H3's image at the H2 top would
    strand images far from their referencing prose.
    """
    sections: list[list[dict]] = [[]]
    for s in segs:
        if _heading_level(s) is not None:
            sections.append([s])
        else:
            sections[-1].append(s)

    out: list[dict] = []
    for sec in sections:
        if not sec:
            continue
        head: list[dict] = []
        images: list[dict] = []
        texts: list[dict] = []
        for s in sec:
            if _heading_level(s) is not None and not head:
                head.append(s)
            elif s.get("type") == "image":
                images.append(s)
            else:
                texts.append(s)
        out.extend(head)
        out.extend(images)
        out.extend(texts)
    return out


def walk_body(body: list) -> list[dict]:
    """Convert GDOC body blocks to a flat list of rendered segments.
    Applies footer-cut (truncate article here) and per-block drop filters.
    Returns list of {"type": "text"|"image", ...}."""
    out: list[dict] = []
    img_order = 0
    for b in body:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        rendered: str | None = None
        if t == "text":
            rendered = _render_text_block(b).strip()
        elif t == "heading":
            rendered = _render_heading_block(b)
        elif t == "callout":
            rendered = _render_callout_block(b)
        elif t in ("list", "numbered-list"):
            rendered = _render_list_block(b)
        elif t == "image":
            out.append({
                "type": "image",
                "filename": b.get("filename") or "",
                "alt": b.get("alt") or "",
                "size": b.get("size") or "",
                "img_order": img_order,
            })
            img_order += 1
            continue
        elif t == "chart":
            # Live grapher chart embed — resolve to a grapher slug.
            url = b.get("url") or ""
            m = GRAPHER_SLUG_RE.search(url)
            out.append({
                "type": "image",
                "url": url,
                "grapher_slug": m.group(1) if m else "",
                "size": b.get("size") or "",
                "alt": "",
                "img_order": img_order,
                "gdoc_kind": "chart",
            })
            img_order += 1
            continue
        else:
            # Skip unknown block types (horizontal-rule, gray-section,
            # prominent-link, sticky-right, scroller, chart-story,
            # narrative-chart, table, key-insights, etc.)
            continue

        if not rendered:
            continue
        if _is_cut_text(rendered):
            break  # truncate article here — everything after is footer
        if _is_drop_text(rendered) or _is_reference_like(rendered):
            continue
        out.append({"type": "text", "value": rendered})
    return out


def match_image_urls(gdoc_segs: list[dict], interleave_segs: list[dict],
                     authors: set[str]) -> list[dict]:
    """Fill in URL/alt for each GDOC image segment by order-matching against
    the interleave segments' non-portrait images. Author portraits and
    'Featured image'/'Thumbnail' images in interleave are skipped so the order
    aligns with GDOC body image blocks."""
    # Build ordered list of content-image URLs from interleave (skipping
    # author portraits, featured images, thumbnails)
    interleave_imgs: list[dict] = []
    for s in interleave_segs:
        if not isinstance(s, dict) or s.get("type") != "image":
            continue
        url = s.get("url") or ""
        # Only CDN-uploaded images are matched to GDOC `image` blocks.
        # Grapher URLs (/grapher/…) are handled separately via GDOC `chart`
        # blocks that carry their own slug.
        if "cdn-cgi/imagedelivery" not in url:
            continue
        alt = (s.get("alt") or "").strip()
        alt_lo = alt.lower()
        # Skip author portraits (alt == author name verbatim)
        if alt in authors:
            continue
        # Skip OWID site-chrome images: featured images, thumbnails,
        # topic-page banners ("X by Our World in Data" over blue background),
        # and draft placeholders.
        if alt_lo.startswith(("featured image", "thumbnail")):
            continue
        if "by our world in data" in alt_lo:
            continue
        if "lighter blue world map" in alt_lo:
            continue
        if alt_lo.startswith("a dark blue background"):
            continue
        if "will fill in when i have the final" in alt_lo:
            continue
        interleave_imgs.append(s)

    # Re-number img_order per type class so we order-match CDN-image blocks
    # against the CDN images in interleave, while `chart` blocks (which already
    # carry their own grapher slug) don't need matching.
    cdn_order = 0
    result = []
    for s in gdoc_segs:
        if s.get("type") != "image":
            result.append(s)
            continue
        if s.get("gdoc_kind") == "chart":
            # chart block already has grapher_slug — pass through
            result.append(s)
            continue
        # Plain image block — match to interleave by CDN order
        if cdn_order < len(interleave_imgs):
            im = interleave_imgs[cdn_order]
            result.append({
                "type": "image",
                "filename": s.get("filename", ""),
                "alt": s.get("alt") or im.get("alt") or "",
                "size": s.get("size", ""),
                "url": im.get("url", ""),
                "img_kind": im.get("img_kind", ""),
            })
            cdn_order += 1
        else:
            continue  # no interleave entry → skip
    return result


# ---------- Main -------------------------------------------------------------

async def main_async(args):
    raw = pl.read_parquet(args.raw).filter(pl.col("kind") == "article")
    interleave = pl.read_parquet(args.interleave).filter(pl.col("kind") == "article")
    grapher = pl.read_parquet(args.grapher)
    grapher_bytes = dict(zip(grapher["slug"].to_list(),
                              grapher["image_bytes"].to_list()))

    interleave_by_slug = {r["slug"]: r for r in interleave.iter_rows(named=True)}
    print(f"articles (raw):        {len(raw):,}", flush=True)
    print(f"articles (interleave): {len(interleave_by_slug):,}", flush=True)
    print(f"grapher cache:         {len(grapher_bytes):,} slugs", flush=True)

    # ---- Phase 1: parse GDOC + render body into segments ----
    plan: list[dict] = []
    cdn_urls_needed: set[str] = set()

    for row in raw.iter_rows(named=True):
        slug = row["slug"]
        gdoc = _extract_gdoc(row["html"])
        if not gdoc:
            continue
        content = gdoc.get("content") or {}
        body = content.get("body") or []
        if not body:
            continue
        authors = content.get("authors") or []

        gdoc_segs = walk_body(body)
        il_row = interleave_by_slug.get(slug)
        il_segs = json.loads(il_row["segments_json"]) if il_row else []
        new_segs = match_image_urls(gdoc_segs, il_segs, set(authors))
        if not new_segs:
            continue
        # Move images to the head of each H2-delimited section so the image
        # precedes the prose that references it. Better for VLM training
        # where image tokens anchor the subsequent text loss.
        new_segs = reorder_by_section(new_segs)

        for s in new_segs:
            if s.get("type") != "image":
                continue
            # Grapher chart: use the slug captured from the GDOC `chart` block
            # directly — don't require an order match against interleave.
            if s.get("gdoc_kind") == "chart" and s.get("grapher_slug"):
                slug_gk = s["grapher_slug"]
                if slug_gk in grapher_bytes:
                    s["source"] = "grapher_cache"
                    continue
                s["source"] = "unresolved_grapher"
                continue
            url = s.get("url") or ""
            m = GRAPHER_SLUG_RE.search(url)
            if m and m.group(1) in grapher_bytes:
                s["source"] = "grapher_cache"
                s["grapher_slug"] = m.group(1)
            elif "cdn-cgi/imagedelivery" in url:
                s["source"] = "cdn_fetch"
                cdn_urls_needed.add(url)
            else:
                s["source"] = "unresolved"

        plan.append({
            "slug": slug,
            "title": content.get("title") or "",
            "subtitle": content.get("subtitle") or "",
            "url": row.get("url", ""),
            "license": row.get("license", "CC-BY-4.0"),
            "authors_json": json.dumps(authors, ensure_ascii=False),
            "dateline": content.get("dateline") or "",
            "excerpt": content.get("excerpt") or "",
            "_segs": new_segs,
        })

    print(f"plan rows: {len(plan):,}  cdn urls to fetch: {len(cdn_urls_needed):,}",
          flush=True)

    # ---- Phase 2: fetch CDN images ----
    bucket = TokenBucket(args.rps)
    sem = asyncio.Semaphore(args.concurrency)
    metrics = {"ok": 0, "err": 0, "404": 0}
    url_to_bytes: dict[str, bytes] = {}

    async def bounded(client, url):
        async with sem:
            b = await fetch(client, bucket, url, metrics)
            if b is not None:
                url_to_bytes[url] = b

    t0 = time.time()
    urls = sorted(cdn_urls_needed)
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=args.concurrency * 2)
    ) as client:
        tasks = [asyncio.create_task(bounded(client, u)) for u in urls]
        for i, task in enumerate(asyncio.as_completed(tasks), start=1):
            await task
            if i % 500 == 0:
                dt = time.time() - t0
                print(f"  [{i:>5}/{len(urls)}] ok={metrics['ok']} "
                      f"err={metrics['err']} 404={metrics['404']}  "
                      f"rate={i/max(dt,1):.1f}/s", flush=True)
    print(f"cdn fetch done in {(time.time()-t0)/60:.1f} min: {metrics}", flush=True)

    # ---- Phase 3: assemble rows with image bytes ----
    out_rows: list[dict] = []
    dropped_img = 0
    for p in plan:
        segs_clean: list[dict] = []
        images_bytes: list[bytes] = []
        images_meta: list[dict] = []
        for s in p["_segs"]:
            if s.get("type") == "image":
                if s.get("source") == "grapher_cache":
                    b = grapher_bytes.get(s["grapher_slug"])
                else:
                    b = url_to_bytes.get(s.get("url", ""))
                if b is None:
                    dropped_img += 1
                    continue
                images_bytes.append(b)
                meta = {k: v for k, v in s.items() if k not in ("img_order",)}
                images_meta.append(meta)
                segs_clean.append({
                    "type": "image",
                    "image_index": len(images_bytes) - 1,
                    "filename": s.get("filename", ""),
                    "alt": s.get("alt", ""),
                })
            else:
                segs_clean.append(s)

        # Trim trailing image segments — if image tokens are masked in the
        # loss, ending with an image wastes context (no next-token loss on
        # those positions). Ensure the final segment is text.
        while segs_clean and segs_clean[-1].get("type") == "image":
            last = segs_clean.pop()
            if last.get("image_index") is not None:
                idx = last["image_index"]
                if idx < len(images_bytes):
                    images_bytes.pop(idx)
                    images_meta.pop(idx)
                    # Any remaining image segments reference by index — shift
                    # any larger image_index down by one.
                    for s2 in segs_clean:
                        if s2.get("type") == "image" and s2.get("image_index", -1) > idx:
                            s2["image_index"] -= 1

        text_segs = [s["value"] for s in segs_clean if s.get("type") == "text"]
        body_text = " ".join(t.strip() for t in text_segs if t.strip())
        word_count = len(body_text.split())
        # Drop topic-page / admin stubs.
        if word_count < args.min_words:
            continue
        out_rows.append({
            "slug": p["slug"],
            "title": p["title"],
            "subtitle": p["subtitle"],
            "dateline": p["dateline"],
            "excerpt": p["excerpt"],
            "url": p["url"],
            "license": p["license"],
            "authors_json": p["authors_json"],
            "body_text": body_text,
            "word_count": word_count,
            "n_segments": len(segs_clean),
            "n_images": len(images_bytes),
            "n_text_chars": sum(len(t) for t in text_segs),
            "segments_json": json.dumps(segs_clean, ensure_ascii=False),
            "images_bytes": images_bytes,
            "images_meta_json": json.dumps(images_meta, ensure_ascii=False),
        })

    print(f"rows with content: {len(out_rows):,}  "
          f"(dropped {len(plan) - len(out_rows)} short/empty; "
          f"{dropped_img} images with missing bytes)", flush=True)

    if not out_rows:
        return
    out = pl.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.out, compression="zstd")
    sz = args.out.stat().st_size / 1e6
    total_img = sum(sum(len(b) for b in r) for r in out["images_bytes"].to_list()) / 1e6
    print(f"wrote {args.out}  ({len(out):,} rows, {sz:.1f} MB, "
          f"{total_img:.1f} MB image bytes inside)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/owid___charts/articles.parquet"))
    ap.add_argument("--interleave", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/owid___charts/articles_interleave.parquet"))
    ap.add_argument("--grapher", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/owid___charts/grapher_charts.parquet"))
    ap.add_argument("--out", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts/articles.parquet"))
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--rps", type=float, default=20.0)
    ap.add_argument("--min-words", type=int, default=200,
                    help="Drop articles whose cleaned body is shorter than this (topic pages).")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
