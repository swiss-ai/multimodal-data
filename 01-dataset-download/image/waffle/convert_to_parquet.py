"""
Convert WAFFLE images + metadata + license info into HF-style parquets.

Splits into two license buckets:
  - permissive/   : Public Domain + CC0 + CC-BY-* (no share-alike)
  - sa/           : CC-BY-SA-* (share-alike obligations)

Drops everything else (unknown licenses, failed API lookups, NC).

Streaming single-pass over original_size_images.tar.gz — never extracts to disk.
Each parquet row: image bytes + filename clue + Wikipedia metadata + license info.

Usage:
    python convert_to_parquet.py \\
        --waffle-dir /capstor/.../tau-vailab___WAFFLE \\
        --output-dir /capstor/.../tau-vailab___WAFFLE/parquet \\
        --num-workers 8
"""
import argparse
import csv
import io
import json
import re
import tarfile
import urllib.parse
from collections import defaultdict
from pathlib import Path

import polars as pl

_FLUSH_BYTES = 200 * 1024 * 1024  # 200 MB per parquet flush

# License classification
_PERMISSIVE_PREFIXES = ("pd", "cc0", "cc-by-1.0", "cc-by-2.0", "cc-by-2.5",
                        "cc-by-3.0", "cc-by-4.0")
_SA_PREFIXES = ("cc-by-sa-",)


def classify_license(license_id: str) -> str:
    """Return 'permissive', 'sa', or 'drop'."""
    if not license_id:
        return "drop"
    lic = license_id.lower().strip()
    # Drop obvious non-commercial markers (none seen in WAFFLE but be safe)
    if "nc" in lic or "nd" in lic:
        return "drop"
    if lic in _PERMISSIVE_PREFIXES or any(lic == p for p in _PERMISSIVE_PREFIXES):
        return "permissive"
    if lic.startswith("cc-by-") and "-sa-" in lic:
        return "sa"
    if lic.startswith("cc-by-sa") or lic.startswith("cc-sa"):
        return "sa"
    if lic in ("pd", "cc0"):
        return "permissive"
    if lic.startswith("cc-by") and "sa" not in lic and "nc" not in lic and "nd" not in lic:
        return "permissive"
    return "drop"


def decode_filename_clue(img_url: str) -> str:
    """Extract human-readable filename from Wikimedia URL.

    e.g. .../Borough_House%2C_West_Side..._%28sheet_3_of_30%29.png
         -> "Borough House, West Side... (sheet 3 of 30)"
    """
    name = img_url.rstrip("/").split("/")[-1]
    name = urllib.parse.unquote(name)
    name = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", name)  # strip extension
    name = name.replace("_", " ")
    return name.strip()


def parse_ocr_texts(raw: str) -> list:
    """Parse the OCR texts column (stored as Python list literal string)."""
    if not raw or raw == "[]":
        return []
    try:
        # safe-ish eval since these are strings written by the dataset authors
        import ast
        v = ast.literal_eval(raw)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    return []


def filter_ocr(ocr_list: list, min_total_chars: int = 10) -> list:
    """Drop garbage OCR (single chars, non-Latin noise, very short)."""
    cleaned = []
    for t in ocr_list:
        t = str(t).strip()
        if len(t) < 2:
            continue
        # Skip if mostly non-alphanumeric
        alpha = sum(1 for c in t if c.isalpha())
        if alpha < 1:
            continue
        cleaned.append(t)
    total_chars = sum(len(t) for t in cleaned)
    if total_chars < min_total_chars:
        return []
    return cleaned


def load_metadata(waffle_dir: Path) -> dict:
    """Load all metadata into a page_id -> {fields} dict."""
    meta = {}

    # Load main dataset.csv (curated)
    with open(waffle_dir / "dataset.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["page_id"]
            meta[pid] = {
                "page_id": pid,
                "img_url": row.get("img_url", ""),
                "img_path": row.get("img_path", ""),
                "building_name": row.get("building_name", ""),
                "building_type": row.get("building_type", ""),
                "high_level_building_type": row.get("high_level_building_type", ""),
                "country": row.get("country", ""),
                "ocr_raw": row.get("ocr_texts", ""),
            }

    # Enrich with raw_dataset.csv columns we care about
    raw_path = waffle_dir / "raw_dataset.csv"
    extra_cols = ["category", "page_content", "city", "state", "region",
                  "wide_clip_score", "narrow_clip_score", "highest_clip_category"]
    if raw_path.exists():
        with open(raw_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("page_id")
                if pid in meta:
                    for col in extra_cols:
                        v = row.get(col, "")
                        if col == "page_content" and v:
                            v = v[:1000]
                        meta[pid][col] = v

    # Load license_map.csv if present
    lic_path = waffle_dir / "license_map.csv"
    if lic_path.exists():
        with open(lic_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("page_id")
                if pid in meta:
                    meta[pid]["license_short"] = row.get("license_short", "")
                    meta[pid]["license_id"] = row.get("license_id", "")
                    meta[pid]["wikimedia_filename"] = row.get("filename", "")

    return meta


def build_path_index(meta: dict) -> dict:
    """Map img_path (relative inside tar) -> page_id.

    Index multiple variants since CSV stores 'data/original_size_images/...'
    but tar entries are 'original_size_images/...' (no 'data/' prefix).
    """
    idx = {}
    for pid, m in meta.items():
        p = m.get("img_path", "")
        if not p:
            continue
        idx[p] = pid
        # Also index without 'data/' prefix (tar omits it)
        if p.startswith("data/"):
            idx[p[5:]] = pid
        # And by basename as last-resort fallback
        from pathlib import Path as _P
        idx[_P(p).name] = pid
    return idx


def build_row(image_bytes: bytes, m: dict) -> dict:
    """Construct a parquet row from image bytes + metadata dict.

    Note: image stored as flat columns (image_bytes, image_path) instead of
    a struct, since polars writes Binary inside Struct with ~5x bloat.
    """
    ocr_clean = filter_ocr(parse_ocr_texts(m.get("ocr_raw", "")))
    fn_clue = decode_filename_clue(m.get("img_url", "")) or m.get("wikimedia_filename", "")
    img_url = m.get("img_url", "")
    page_id = m["page_id"]
    wikimedia_url = f"https://commons.wikimedia.org/?curid={page_id}" if page_id else ""
    return {
        "image_bytes": image_bytes,
        "image_path": m.get("img_path", ""),
        "page_id": page_id,
        "img_url": img_url,
        "wikimedia_page_url": wikimedia_url,
        "wikimedia_filename": m.get("wikimedia_filename", ""),
        "building_name": m.get("building_name", ""),
        "building_type": m.get("building_type", ""),
        "high_level_building_type": m.get("high_level_building_type", ""),
        "country": m.get("country", ""),
        "city": m.get("city", ""),
        "state": m.get("state", ""),
        "region": m.get("region", ""),
        "category": m.get("category", ""),
        "page_content": m.get("page_content", ""),
        "filename_clue": fn_clue,
        "ocr_texts_clean": ocr_clean,
        "wide_clip_score": m.get("wide_clip_score", ""),
        "narrow_clip_score": m.get("narrow_clip_score", ""),
        "license_short": m.get("license_short", ""),
        "license_id": m.get("license_id", ""),
    }


def flush_bucket(rows: list, out_dir: Path, bucket: str, idx: int) -> str:
    out_path = out_dir / bucket / f"train-{idx:05d}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use pyarrow directly — polars.write_parquet bloats Binary columns ~5x.
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pl.DataFrame(rows).to_arrow()
    # snappy: fast decompression for image data loaders (JPEG already compressed)
    pq.write_table(table, out_path, compression="snappy")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waffle-dir", required=True,
                        help="Dir with dataset.csv, raw_dataset.csv, license_map.csv, original_size_images.tar.gz")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    waffle_dir = Path(args.waffle_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata...", flush=True)
    meta = load_metadata(waffle_dir)
    print(f"  {len(meta)} samples in dataset.csv", flush=True)

    # Pre-classify licenses
    cat_counts = defaultdict(int)
    for pid, m in meta.items():
        cat = classify_license(m.get("license_id", ""))
        m["__bucket"] = cat
        cat_counts[cat] += 1
    print(f"License classification:")
    for cat, c in cat_counts.items():
        print(f"  {cat}: {c}")

    path_index = build_path_index(meta)
    print(f"Path index built: {len(path_index)} entries", flush=True)

    # Stream tar.gz, bucket rows, flush per bucket when threshold hit
    tar_path = waffle_dir / "original_size_images.tar.gz"
    print(f"\nStreaming {tar_path}...", flush=True)

    buffers = {"permissive": [], "sa": []}
    buffer_bytes = {"permissive": 0, "sa": 0}
    parquet_idx = {"permissive": 0, "sa": 0}
    total_written = {"permissive": 0, "sa": 0}
    skipped_no_meta = 0
    skipped_dropped = 0
    seen = 0

    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            # tar entries may have leading "./"; normalize
            n = name.lstrip("./")
            if not (n.endswith(".jpg") or n.endswith(".png") or n.endswith(".jpeg")):
                continue
            seen += 1

            # Find matching metadata via img_path
            pid = path_index.get(n) or path_index.get("./" + n) or path_index.get(name)
            if pid is None:
                # Try to match by basename
                base = Path(n).name
                # not indexed by basename; just skip
                skipped_no_meta += 1
                continue

            m = meta.get(pid)
            if m is None:
                skipped_no_meta += 1
                continue
            bucket = m["__bucket"]
            if bucket == "drop":
                skipped_dropped += 1
                continue

            try:
                f = tf.extractfile(member)
                image_bytes = f.read() if f is not None else None
                if not image_bytes:
                    continue
            except Exception:
                continue

            row = build_row(image_bytes, m)
            buffers[bucket].append(row)
            buffer_bytes[bucket] += len(image_bytes)

            if buffer_bytes[bucket] >= _FLUSH_BYTES:
                out_path = flush_bucket(buffers[bucket], out_dir, bucket, parquet_idx[bucket])
                total_written[bucket] += len(buffers[bucket])
                print(f"  [{bucket}] wrote {Path(out_path).name} "
                      f"({len(buffers[bucket])} rows, {buffer_bytes[bucket]/1e6:.1f} MB) "
                      f"running total: {total_written[bucket]}", flush=True)
                buffers[bucket] = []
                buffer_bytes[bucket] = 0
                parquet_idx[bucket] += 1

    # Final flush
    for bucket in ("permissive", "sa"):
        if buffers[bucket]:
            out_path = flush_bucket(buffers[bucket], out_dir, bucket, parquet_idx[bucket])
            total_written[bucket] += len(buffers[bucket])
            print(f"  [{bucket}] final {Path(out_path).name} "
                  f"({len(buffers[bucket])} rows)", flush=True)

    print(f"\n=== Summary ===")
    print(f"Tar entries seen (images): {seen}")
    print(f"Skipped — no metadata:    {skipped_no_meta}")
    print(f"Skipped — license drop:   {skipped_dropped}")
    print(f"Written — permissive:     {total_written['permissive']}")
    print(f"Written — sa:             {total_written['sa']}")


if __name__ == "__main__":
    main()
