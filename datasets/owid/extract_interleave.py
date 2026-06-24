"""Convert OWID articles HTML → interleave (text + image-ref) sequences.

Reads ``articles.parquet`` (raw HTML produced by ``download_articles.py``)
and writes ``articles_interleave.parquet`` where each row is a single
interleave document: ordered list of ``{type: "text"|"image", ...}``
segments, in DOM order.

Output schema:
    slug:           str
    kind:           str ("article" | "data_insight")
    url:            str
    title:          str
    n_segments:     int
    n_images:       int
    n_text_chars:   int
    segments_json:  str  (JSON array of segments)

Each segment is one of:
    {"type": "text", "value": "<markdown-ish text>"}
    {"type": "image", "url": "<absolute URL>", "alt": "<alt text>"}

Usage:
    python extract_interleave.py \
        --in  /capstor/.../raw/cooldown/owid___charts/articles.parquet \
        --out /capstor/.../raw/cooldown/owid___charts/articles_interleave.parquet
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Iterator, List
from urllib.parse import urljoin

import polars as pl
from selectolax.parser import HTMLParser, Node

# Tags we never want to descend into (navigation, scripts, etc.).
_SKIP_TAGS = {
    "script", "style", "noscript", "svg", "head", "meta", "link",
    "nav", "footer", "header", "aside", "form", "button", "iframe",
}

# Block-level tags whose text contributes a paragraph break.
_BLOCK_TAGS = {
    "p", "div", "section", "article", "blockquote", "pre", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "ul", "ol", "dl", "dt", "dd", "table", "tr", "td", "th",
    "figure", "figcaption", "details", "summary",
}

_HEADER_PREFIX = {f"h{i}": "#" * i for i in range(1, 7)}


@dataclass
class _Segment:
    kind: str  # "text" or "image"
    payload: Any  # str for text, dict for image


def _direct_children(node: Node) -> Iterator[Node]:
    """Yield direct children (DOM order). Selectolax's ``iter`` is flat, so
    we walk via ``node.child`` and ``.next`` to get one level only."""
    child = node.child
    while child is not None:
        yield child
        child = child.next


def _walk(node: Node, base_url: str) -> Iterator[_Segment]:
    """DFS through the HTML tree, emitting text and image segments in DOM order.

    Strategy: at each node, walk *direct* children only and recurse into block
    elements; collapse inline-element text into the surrounding paragraph.
    """
    tag = (node.tag or "").lower()
    if tag in _SKIP_TAGS:
        return

    # <img>: emit image segment, no descent
    if tag == "img":
        src = node.attributes.get("src") or ""
        if not src:
            return
        absolute = urljoin(base_url, src.strip())
        alt = (node.attributes.get("alt") or "").strip()
        # Classify so downstream can filter:
        #   "grapher": owid grapher PNG (canonical chart)
        #   "cdn":     Cloudflare image-delivery (charts, hero images, author photos)
        #   "other":   anything else
        if "/grapher/" in absolute:
            kind = "grapher"
        elif "imagedelivery" in absolute or "cdn-cgi" in absolute:
            kind = "cdn"
        else:
            kind = "other"
        yield _Segment("image", {"url": absolute, "alt": alt, "img_kind": kind})
        return

    text_buf: List[str] = []

    def flush_text() -> Iterator[_Segment]:
        if not text_buf:
            return
        joined = "".join(text_buf).strip()
        text_buf.clear()
        if joined:
            yield _Segment("text", joined)

    if tag in _HEADER_PREFIX:
        text_buf.append(_HEADER_PREFIX[tag] + " ")

    for child in _direct_children(node):
        ctag = (child.tag or "").lower()
        if ctag == "-text":
            text_buf.append(child.text(strip=False) or "")
            continue
        if ctag == "br":
            text_buf.append("\n")
            continue
        if ctag in _SKIP_TAGS:
            continue
        if ctag == "img":
            yield from flush_text()
            yield from _walk(child, base_url)
            continue
        if ctag in _BLOCK_TAGS:
            # Block element: flush current text run, recurse to get nested
            # text+images in their own segments
            yield from flush_text()
            yield from _walk(child, base_url)
            continue
        # Inline element (a, span, em, strong, code, ...):
        # collect its text into the current run, but recurse for any nested
        # images / blocks inside.
        nested = list(_walk(child, base_url))
        for n in nested:
            if n.kind == "text":
                text_buf.append(" " + n.payload + " ")
            else:
                yield from flush_text()
                yield n

    yield from flush_text()


def _normalize_text(text: str) -> str:
    """Collapse runs of whitespace, keep paragraph-style line breaks."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def html_to_interleave(html: str, base_url: str) -> List[dict]:
    """Parse HTML and return a list of interleave segments in DOM order.

    Walks only the main article body (``div#owid-document-root``) to avoid
    contamination from related-article sidebars, footer recommendations,
    nav, etc. Falls back to ``<body>`` if the OWID-specific root is absent.
    """
    tree = HTMLParser(html)
    root = tree.css_first("div#owid-document-root") or tree.body or tree.root
    if root is None:
        return []
    body = root

    segments: List[dict] = []
    pending_text: List[str] = []

    def flush_text():
        if not pending_text:
            return
        joined = "\n\n".join(p for p in pending_text if p.strip())
        normalized = _normalize_text(joined)
        pending_text.clear()
        if normalized:
            # Merge adjacent text segments
            if segments and segments[-1]["type"] == "text":
                segments[-1]["value"] = _normalize_text(
                    segments[-1]["value"] + "\n\n" + normalized
                )
            else:
                segments.append({"type": "text", "value": normalized})

    for seg in _walk(body, base_url):
        if seg.kind == "text":
            pending_text.append(seg.payload)
        else:
            flush_text()
            segments.append({"type": "image", **seg.payload})

    flush_text()
    return segments


def _extract_title(html: str) -> str:
    tree = HTMLParser(html)
    if tree.css_first("h1"):
        return (tree.css_first("h1").text() or "").strip()
    if tree.css_first("title"):
        return (tree.css_first("title").text() or "").strip()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True, help="Input articles.parquet")
    ap.add_argument("--out", dest="out_path", required=True, help="Output articles_interleave.parquet")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N rows (for debugging)")
    ap.add_argument("--kind", action="append", default=None,
                    help="Filter to specific kind(s): 'article' or 'data_insight'. Default = all.")
    args = ap.parse_args()

    print(f"reading {args.in_path}")
    df = pl.read_parquet(args.in_path)
    if args.kind:
        df = df.filter(pl.col("kind").is_in(args.kind))
    if args.limit:
        df = df.head(args.limit)

    print(f"processing {len(df):,} rows")

    rows = []
    for r in df.iter_rows(named=True):
        if not r.get("html"):
            continue
        url = r["url"]
        try:
            segments = html_to_interleave(r["html"], base_url=url)
        except Exception as e:
            print(f"  WARN: {r['slug']}: {e}")
            continue
        title = _extract_title(r["html"])
        n_text = sum(1 for s in segments if s["type"] == "text")
        n_img = sum(1 for s in segments if s["type"] == "image")
        n_chars = sum(len(s["value"]) for s in segments if s["type"] == "text")
        rows.append({
            "slug": r["slug"],
            "kind": r["kind"],
            "url": r["url"],
            "title": title,
            "n_segments": n_text + n_img,
            "n_text_segments": n_text,
            "n_images": n_img,
            "n_text_chars": n_chars,
            "segments_json": json.dumps(segments, ensure_ascii=False),
        })

    out_df = pl.DataFrame(rows)
    out_df.write_parquet(args.out_path)
    print(f"wrote {len(rows):,} interleave docs → {args.out_path}")
    print(f"  total images: {out_df['n_images'].sum():,}")
    print(f"  total text chars: {out_df['n_text_chars'].sum():,}")
    print(f"  per-doc averages: text_segments={out_df['n_text_segments'].mean():.1f}, "
          f"images={out_df['n_images'].mean():.1f}, "
          f"text_chars={out_df['n_text_chars'].mean():.0f}")


if __name__ == "__main__":
    main()
