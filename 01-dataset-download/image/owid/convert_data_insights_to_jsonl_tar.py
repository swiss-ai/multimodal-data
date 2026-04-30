"""Convert ``data_insights.parquet`` → jsonl_tar interleave format.

Each OWID data insight becomes one interleave document with structure:

    # <title>

    <body_text>

    ![<image_alt>](<slug>.png)

    <image_alt>

The ``pin200m`` parser splits this into [text, image, text] segments at
tokenization time (splits on ``![...](...)`` markdown refs).

Output layout (matches other jsonl_tar datasets):

    <out_dir>/
      owid_data_insights.jsonl
      owid_data_insights-000.tar  (416 PNGs, one per slug)

Usage:
    python convert_data_insights_to_jsonl_tar.py \
        --in  /capstor/.../processed/owid___charts/data_insights.parquet \
        --out /capstor/.../processed/owid___charts/data_insights_jsonl_tar
"""
from __future__ import annotations

import argparse
import json
import tarfile
from io import BytesIO
from pathlib import Path

import polars as pl


def build_md(title: str, body: str, alt: str, image_ref: str) -> str:
    """Assemble a data insight into a single markdown document the pin200m
    parser can split into ``[image, text(title+body), text(alt)]``.

    Order: image first, then body, then alt — so training conditions
    text generation on the chart (captioning-style) rather than the
    reverse. Title and body are joined with a single newline so pin200m's
    ``pre_segment_text`` (which splits on ``\\n\\n+``) keeps them in one
    text segment, while the blank line before alt makes alt its own
    segment.

    Note on the inline alt: some OWID alts contain ``]`` (e.g.
    ``[friends are]``) which breaks pin200m's ``![alt](src)`` regex,
    since alt capture is ``[^\\]]*``. We use a neutral ``![chart](src)``
    placeholder so the image ref parses cleanly; the full alt text is
    preserved verbatim as the second text segment below.
    """
    body = (body or "").strip()
    alt = (alt or "").strip()
    title = (title or "").strip()

    parts: list[str] = [f"![chart]({image_ref})"]
    title_body_parts = []
    if title:
        title_body_parts.append(f"# {title}")
    if body:
        title_body_parts.append(body)
    if title_body_parts:
        parts.append("\n".join(title_body_parts))
    if alt:
        parts.append(alt)

    return "\n\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_dir", required=True)
    ap.add_argument("--shard-name", default="owid_data_insights")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.shard_name}.jsonl"
    tar_path = out_dir / f"{args.shard_name}-000.tar"

    df = pl.read_parquet(args.in_path)
    print(f"read {len(df):,} rows from {args.in_path}")

    # Guard against duplicate slugs (tar member names must be unique)
    seen: set[str] = set()
    n_written = 0
    n_skipped_dup = 0
    n_skipped_no_img = 0

    with tarfile.open(tar_path, "w") as tar, jsonl_path.open("w") as jf:
        for r in df.iter_rows(named=True):
            slug = (r.get("slug") or "").strip()
            img_bytes = r.get("image_bytes")
            if not slug or not img_bytes:
                n_skipped_no_img += 1
                continue
            if slug in seen:
                n_skipped_dup += 1
                continue
            seen.add(slug)

            # Tar member: image/<slug>.png. The ``image/`` prefix matches the
            # default ``local_image_prefixes`` that the pin200m parser uses to
            # recognize in-document image refs (see parsers/common.py).
            member_name = f"image/{slug}.png"
            info = tarfile.TarInfo(name=member_name)
            info.size = len(img_bytes)
            info.mtime = 0
            tar.addfile(info, BytesIO(img_bytes))

            # JSONL row: md field drives the pin200m parser; keep extras
            # so downstream consumers can inspect provenance.
            md = build_md(
                title=r.get("title", ""),
                body=r.get("body_text", ""),
                alt=r.get("image_alt", ""),
                image_ref=member_name,
            )
            row_out = {
                "md": md,
                "slug": slug,
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "license": r.get("license") or "",
                "authors_json": r.get("authors_json") or "[]",
                "word_count": int(r.get("word_count") or 0),
            }
            jf.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"wrote {n_written:,} docs -> {jsonl_path}")
    print(f"wrote tar -> {tar_path}")
    if n_skipped_dup:
        print(f"  skipped {n_skipped_dup} duplicate slugs")
    if n_skipped_no_img:
        print(f"  skipped {n_skipped_no_img} rows with no image")


if __name__ == "__main__":
    main()
