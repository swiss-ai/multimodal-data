"""Parse raw OWID article HTML into clean (slug, title, authors, body_text,
embedded_chart_slugs).

Uses the ``window._OWID_GDOC_PROPS = {...}`` JSON blob embedded in each page,
so we avoid HTML wrangling entirely.

Input:  raw/cooldown/owid___charts/articles.parquet
Output: processed/owid___charts/articles_clean.parquet
"""
import argparse
import json
import re
from pathlib import Path

import polars as pl


_GDOC_START_RE = re.compile(r"window\._OWID_GDOC_PROPS\s*=\s*\{")


def _extract_gdoc(html: str) -> dict | None:
    """Extract the JSON object from ``window._OWID_GDOC_PROPS = {...};``.

    Regex on its own can't reliably match ~30kB of JSON that contains escaped
    braces in strings; walk the text with a brace counter instead, skipping
    characters inside quoted strings.
    """
    m = _GDOC_START_RE.search(html)
    if not m:
        return None
    start = m.end() - 1  # include the opening `{`
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(html)):
        ch = html[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _render_spans(spans: list) -> str:
    """Flatten OWID span-tree (span-simple-text, span-link, span-bold, ...) to
    plain text. Spans can nest via `children`."""
    parts = []
    for s in spans or []:
        if not isinstance(s, dict):
            continue
        t = s.get("spanType")
        if t == "span-simple-text":
            parts.append(s.get("text", ""))
        elif t == "span-newline":
            parts.append("\n")
        else:
            # bold/italic/link/etc carry children
            parts.append(_render_spans(s.get("children") or []))
    return "".join(parts)


def _render_block(b: dict) -> tuple[str, list[str]]:
    """Return (text, chart_slugs) for one body block."""
    if not isinstance(b, dict):
        return "", []
    kind = b.get("type")
    if kind == "text":
        return _render_spans(b.get("value") or []), []
    if kind == "heading":
        txt = _render_spans(b.get("text") or b.get("value") or [])
        return txt, []
    if kind == "list" or kind == "numbered-list":
        items = []
        for item in b.get("items") or []:
            items.append(_render_spans(item.get("value") or []))
        return "\n".join(f"- {i}" for i in items), []
    if kind == "callout":
        children = b.get("text") or b.get("value") or []
        if isinstance(children, list):
            return "\n".join(_render_spans(c.get("value") or []) if isinstance(c, dict) else ""
                             for c in children), []
    if kind in ("chart", "chart-story", "image", "video", "gray-section"):
        slug = b.get("url") or b.get("slug") or ""
        m = re.search(r"/grapher/([^/?#]+)", slug)
        return "", [m.group(1)] if m else []
    # Unknown block types are silently skipped (we don't want to pollute text).
    return "", []


_CTA_RE = re.compile(
    r"^(Explore|Read|See|Learn|Discover|Find|Browse)\b.*(?:→|↗)\s*$",
    re.IGNORECASE,
)

# Tail markers that indicate the end of the article body — everything from the
# match onward is structural/footer content (acknowledgments, related-article
# pull-outs, citation boilerplate), not article prose. Matching is greedy over
# the first occurrence so we catch both "Acknowledgments:" and "Acknowledgments "
# (which OWID uses interchangeably as a section heading).
_FOOTER_CUT_RE = re.compile(
    r"\s*(?:"
    r"Acknowledgments?\b:?"        # with or without colon
    r"|Acknowledgements?\b:?"      # British spelling
    r"|I would like to thank\s+[A-Z]"
    r"|We thank\s+[A-Z]"
    r"|Many thanks to\s+[A-Z]"
    r"|Thanks to\s+[A-Z][\w\.\-']+(?:\s+[A-Z][\w\.\-']+)*"  # "Thanks to <Name> [<Lastname>]"
    r"\s*(?:,\s*[A-Z][\w\.\-']+(?:\s+[A-Z][\w\.\-']+)*)*"    # ", <More Names>"
    r"\s*(?:and\s+[A-Z][\w\.\-']+(?:\s+[A-Z][\w\.\-']+)*)?"  # " and <Final Name>"
    r"\s+for\s+(?:their|editorial|helpful|valuable|the)"     # "for their/editorial/…"
    r"|Keep reading on Our World in Data"
    r"|Explore more research and data on"
    r"|Cite this work"
    r"|Reuse this work freely"
    r")",
    re.IGNORECASE,
)


def _is_cta_paragraph(text: str) -> bool:
    """True if `text` is a short call-to-action tail like
    'Explore this data →' or 'Read more on renewable energy →'."""
    t = text.strip()
    if not t:
        return False
    if len(t.split()) > 30:
        return False
    return bool(_CTA_RE.match(t)) or t in {"Read more →", "Read more", "Explore the data →"}


def parse_article(html: str) -> dict | None:
    gdoc = _extract_gdoc(html)
    if not gdoc:
        return None
    content = gdoc.get("content") or {}
    title = content.get("title") or ""
    authors = content.get("authors") or []
    body = content.get("body") or []

    paragraphs = []
    charts: list[str] = []
    for block in body:
        text, slugs = _render_block(block)
        if text:
            paragraphs.append(text)
        charts.extend(slugs)

    # Drop trailing CTA paragraphs (may be more than one).
    while paragraphs and _is_cta_paragraph(paragraphs[-1]):
        paragraphs.pop()

    # Flatten into a single paragraph — separator is a plain space so the
    # downstream tokenizer sees continuous prose instead of paragraph breaks.
    body_text = " ".join(p.strip() for p in paragraphs if p.strip())

    # Truncate at footer markers that indicate the end of the article body:
    # acknowledgments, related-links pull-out sections, citation boilerplate.
    # These add noise to training — they're structural site chrome, not content.
    body_text = _FOOTER_CUT_RE.split(body_text, maxsplit=1)[0].rstrip()
    return {
        "title": title,
        "authors": authors,
        "body_text": body_text,
        "word_count": len(body_text.split()),
        "chart_slugs": list(dict.fromkeys(charts)),  # dedupe, preserve order
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/owid___charts/articles.parquet"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts"))
    args = ap.parse_args()

    df = pl.read_parquet(args.in_parquet)
    print(f"input: {len(df):,} rows", flush=True)

    out_rows = []
    n_ok, n_fail = 0, 0
    for row in df.iter_rows(named=True):
        parsed = parse_article(row["html"])
        if parsed is None:
            n_fail += 1
            continue
        n_ok += 1
        out_rows.append({
            "slug": row["slug"],
            "kind": row["kind"],
            "url": row["url"],
            "license": row.get("license", "CC-BY-4.0"),
            "title": parsed["title"],
            "authors_json": json.dumps(parsed["authors"], ensure_ascii=False),
            "body_text": parsed["body_text"],
            "word_count": parsed["word_count"],
            "chart_slugs_json": json.dumps(parsed["chart_slugs"]),
        })

    print(f"parsed ok: {n_ok:,}  failed: {n_fail:,}", flush=True)
    if not out_rows:
        return

    out = pl.DataFrame(out_rows)
    # Word-count stats
    wcs = out["word_count"].to_list()
    print(f"word_count: min={min(wcs)} median={sorted(wcs)[len(wcs)//2]} "
          f"max={max(wcs)} avg={sum(wcs)/len(wcs):.0f}", flush=True)
    # Kind breakdown
    by_kind = out.group_by("kind").agg(
        pl.len().alias("n"),
        pl.col("word_count").mean().round(0).alias("mean_wc"),
    )
    print(by_kind)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind, outname in [("data_insight", "data_insights_clean.parquet"),
                          ("article", "articles_clean.parquet")]:
        sub = out.filter(pl.col("kind") == kind)
        p = args.out_dir / outname
        sub.write_parquet(p, compression="zstd")
        print(f"  wrote {p}  ({len(sub):,} rows)", flush=True)


if __name__ == "__main__":
    main()
