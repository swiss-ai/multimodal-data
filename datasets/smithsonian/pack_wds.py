#!/usr/bin/env python3
"""
Phase 4 — Pack into WebDataset .tar shards.

Reads caption Parquet files and writes:
  {OUT_ROOT}/{tier_path}/{000000..XXXXXX}.tar

Each shard tar contains per sample:
  {key}.jpg    highest-quality image
  {key}.txt    training-ready caption
  {key}.json   full metadata / grounding prompt
"""

import hashlib
import io
import json
import logging
import sys
import tarfile
from multiprocessing import Pool, cpu_count
from pathlib import Path


def _to_list(val):
    """Safely convert numpy arrays / None to plain Python list."""
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return [val] if val else []
    try:
        return list(val)
    except Exception:
        return []


def _to_dict(val):
    """Safely convert to plain Python dict."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    try:
        return dict(val)
    except Exception:
        return {}


import pandas as pd
from PIL import Image
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
CAPTION_DIR = Path("/tmp/toolbox/smithsonian/data/captions")
OUT_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian")
LOG_DIR = Path("/tmp/toolbox/smithsonian/data")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "pack_webdataset.log"),
    ],
)
log = logging.getLogger("pack_wds")

# ── Config ─────────────────────────────────────────────────────────────────────
SHARD_SIZE = {
    "tier1": 500,
    "tier2": 500,
    "tier3": 1000,
    "tier4": 200,
}
MIN_DIM = 224  # discard images smaller than this on either axis
MAX_WORKERS = min(256, cpu_count())

# Disable PIL decompression bomb check — large natural history scans are legitimate
from PIL import Image as _PIL_Image

_PIL_Image.MAX_IMAGE_PIXELS = None


# ── Image loading & validation ─────────────────────────────────────────────────


def load_image_bytes(image_path: str) -> tuple[bytes | None, int, int]:
    """
    Load image, validate minimum dimensions, return (jpeg_bytes, width, height).
    Returns (None, 0, 0) if invalid or too small.
    """
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            if w < MIN_DIM or h < MIN_DIM:
                return None, w, h
            # Re-encode as JPEG to normalise format (some may be TIFF/PNG)
            if img.format == "JPEG":
                with open(image_path, "rb") as fh:
                    return fh.read(), w, h
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=92)
            return buf.getvalue(), w, h
    except Exception:
        return None, 0, 0


# ── Key generation ─────────────────────────────────────────────────────────────


def make_key(row: dict) -> str:
    """Produce a URL-safe, unique key for this sample."""
    raw = f"{row.get('collection', '')}_{row.get('primary_idsId', row.get('record_id', ''))}"
    # Sanitise: keep alphanumerics, hyphens, underscores, dots
    import re

    safe = re.sub(r"[^A-Za-z0-9._\-]", "_", raw)
    # If still too long (> 200 chars), hash the tail
    if len(safe) > 200:
        h = hashlib.sha1(safe.encode()).hexdigest()[:8]
        safe = safe[:190] + "_" + h
    return safe


# ── Shard writer ───────────────────────────────────────────────────────────────


def write_shards(tier_path: str, rows: list[dict], shard_start: int = 0) -> dict:
    """
    Write rows for one tier_path into shards starting at shard_start.
    Multiple workers can write non-overlapping shard ranges for the same tier.
    Returns stats dict.
    """
    out_dir = OUT_ROOT / tier_path
    out_dir.mkdir(parents=True, exist_ok=True)

    tier = tier_path.split("/")[0]
    max_count = SHARD_SIZE.get(tier, 500)
    shard_idx = shard_start
    n_written = 0
    n_skipped = 0
    tar = None
    tar_path = None

    # Lazy shard open: only create the file when there is a valid row to write.
    # This prevents an empty shard being created (and then deleted) at chunk
    # boundaries, which would race with the next chunk's worker opening the
    # same shard index.
    def ensure_shard_open():
        nonlocal tar, tar_path
        if tar is None:
            p = out_dir / f"{shard_idx:06d}.tar"
            tar_path = p
            tar = tarfile.open(p, "w")

    def close_shard():
        nonlocal tar, tar_path
        if tar is not None:
            tar.close()
            tar = None
            tar_path = None

    shard_count = 0

    for row in rows:
        img_path = row.get("image_path", "")
        if not img_path or not Path(img_path).exists():
            n_skipped += 1
            continue

        img_bytes, w, h = load_image_bytes(img_path)
        if img_bytes is None:
            n_skipped += 1
            continue

        key = make_key(row)
        ensure_shard_open()

        # ── Caption text ──────────────────────────────────────────────────────
        caption = (row.get("caption_text") or "").strip()
        if not caption:
            n_skipped += 1
            continue

        # ── Metadata JSON ─────────────────────────────────────────────────────
        notes_raw = _to_list(row.get("notes_raw"))
        topics = _to_list(row.get("topics"))
        all_ids = _to_list(row.get("all_idsIds"))
        renders = _to_dict(row.get("renders"))

        meta = {
            "key": key,
            "collection": row.get("collection", "") or "",
            "tier": row.get("tier", tier_path.split("/")[0]) or "",
            "tier_path": tier_path,
            "record_id": row.get("record_id", "") or "",
            "type": row.get("type", "") or "",
            "title": row.get("title", "") or "",
            "date": row.get("date", "") or "",
            "creator": row.get("creator", "") or "",
            "medium": row.get("medium", "") or "",
            "place": row.get("place", "") or "",
            "object_type": row.get("object_type", "") or "",
            "topics": topics,
            "credit_line": row.get("credit_line", "") or "",
            "data_source": row.get("data_source", "") or "",
            "taxonomic_name": row.get("taxonomic_name", "") or "",
            "notes_raw": notes_raw,
            "scopecontent": row.get("scopecontent", "") or "",
            "alt_text": row.get("alt_text", "") or "",
            "ext_descr": row.get("ext_descr", "") or "",
            "caption_source": row.get("caption_source", "") or "",
            "grounding_prompt": row.get("grounding_prompt", "") or "",
            "guid": row.get("guid", "") or "",
            "license": "CC0",
            "image_path": img_path,
            "image_width": w,
            "image_height": h,
            "all_idsIds": all_ids,
            "renders": renders,
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        caption_bytes = caption.encode("utf-8")

        # ── Write to tar ──────────────────────────────────────────────────────
        def add(name, data):
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add(f"{key}.jpg", img_bytes)
        add(f"{key}.txt", caption_bytes)
        add(f"{key}.json", meta_bytes)

        n_written += 1
        shard_count += 1

        if shard_count >= max_count:
            close_shard()
            shard_idx += 1
            shard_count = 0
            # Do NOT pre-open the next shard here; lazy open on next valid row.

    close_shard()  # flush the final partial shard (if any)
    return {
        "tier_path": tier_path,
        "written": n_written,
        "skipped": n_skipped,
        "shards": shard_idx - shard_start + (1 if shard_count > 0 else 0),
        "shard_start": shard_start,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def write_shards_from_file(args):
    """Worker entry point: read a pre-sliced chunk parquet then pack shards.
    Each worker reads only its own small chunk file — no large parquet in every worker.
    """
    tier_path, chunk_parquet, shard_start = args
    df = pd.read_parquet(chunk_parquet)
    rows = df.to_dict("records")
    return write_shards(tier_path, rows, shard_start=shard_start)


def main():
    caption_files = [
        CAPTION_DIR / "captions_art.parquet",
        CAPTION_DIR / "captions_nmnh.parquet",
        CAPTION_DIR / "captions_3d.parquet",
    ]

    # ── Step 1: split each caption parquet by tier_path into small temp files ──
    TIER_DIR = LOG_DIR / "tier_splits"
    TIER_DIR.mkdir(parents=True, exist_ok=True)

    tier_row_counts: dict[str, int] = {}

    # Check if all splits already exist (re-entrant: skip step 1 if done)
    all_splits_exist = TIER_DIR.exists() and len(list(TIER_DIR.glob("*.parquet"))) > 0

    if all_splits_exist:
        log.info(f"Tier splits already exist in {TIER_DIR} — skipping Step 1")
        for p in sorted(TIER_DIR.glob("*.parquet")):
            tp = p.stem.replace("__", "/")
            n = len(pd.read_parquet(p))
            tier_row_counts[tp] = n
    else:
        for cf in caption_files:
            if not cf.exists():
                log.warning(f"Caption file not found, skipping: {cf}")
                continue
            log.info(f"Loading {cf}…")
            df = pd.read_parquet(cf)
            log.info(f"  {len(df):,} records")

            for tp, grp in df.groupby("tier_path"):
                safe_tp = str(tp).replace("/", "__")
                dest = TIER_DIR / f"{safe_tp}.parquet"
                grp.to_parquet(dest, index=False)
                tier_row_counts[str(tp)] = len(grp)

    log.info(f"\nTier paths ({len(tier_row_counts)} total):")
    total = 0
    for tp in sorted(tier_row_counts):
        log.info(f"  {tp}: {tier_row_counts[tp]:,}")
        total += tier_row_counts[tp]
    log.info(f"  TOTAL: {total:,}")

    # ── Step 2: pre-slice tier parquets into per-chunk files ─────────────────
    # Each worker reads its own small chunk file to avoid loading the full
    # (potentially 2M-row) parquet in every worker process simultaneously.
    import math

    TOTAL_WORKERS = min(MAX_WORKERS, 128)  # 128 workers × ~300 MB ≈ 38 GB RAM

    CHUNK_DIR = LOG_DIR / "tier_chunks"
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which tiers need work and how many rows remain
    todo: dict[str, int] = {}
    done_shards: dict[str, int] = {}

    for tp, n_rows in tier_row_counts.items():
        tier = tp.split("/")[0]
        shard_sz = SHARD_SIZE.get(tier, 500)
        out_dir = OUT_ROOT / tp
        n_done = len(list(out_dir.glob("*.tar"))) if out_dir.exists() else 0
        expected = math.ceil(n_rows / shard_sz)
        if n_done >= expected:
            log.info(f"  {tp}: already complete ({n_done} shards) — skipping")
        else:
            todo[tp] = n_rows
            done_shards[tp] = n_done

    if not todo:
        log.info("All tiers already packed. Nothing to do.")
        return

    total_rows = sum(todo.values())
    pack_tasks: list[tuple] = []

    log.info(f"\nPre-slicing {len(todo)} tiers into chunk parquets…")
    for tp, n_rows in sorted(todo.items()):
        tier_parquet = TIER_DIR / f"{tp.replace('/', '__')}.parquet"
        tier = tp.split("/")[0]
        shard_sz = SHARD_SIZE.get(tier, 500)
        n_done = done_shards[tp]
        row_offset = n_done * shard_sz

        rows_remaining = n_rows - row_offset
        if rows_remaining <= 0:
            continue

        # Proportional worker count; chunk size = multiple of shard_sz
        fraction = rows_remaining / total_rows
        n_workers_tier = max(1, round(fraction * TOTAL_WORKERS))
        raw_chunk = rows_remaining // n_workers_tier
        rows_per_chunk = max(shard_sz, (raw_chunk // shard_sz) * shard_sz)

        # Load tier parquet once, write per-chunk files, then free
        safe_tp = tp.replace("/", "__")
        df = pd.read_parquet(tier_parquet)
        chunk_idx = 0
        for start in range(row_offset, n_rows, rows_per_chunk):
            end = min(start + rows_per_chunk, n_rows)
            shard_start = n_done + chunk_idx * (rows_per_chunk // shard_sz)
            chunk_path = CHUNK_DIR / f"{safe_tp}_{chunk_idx:04d}.parquet"
            if not chunk_path.exists():
                df.iloc[start:end].to_parquet(chunk_path, index=False)
            pack_tasks.append((tp, chunk_path, shard_start))
            chunk_idx += 1
        del df  # free tier parquet before loading the next
        log.info(f"  {tp}: {chunk_idx} chunks, {rows_per_chunk:,} rows/chunk")

    n_workers = min(TOTAL_WORKERS, len(pack_tasks))
    log.info(f"\nPacking {len(pack_tasks)} chunks across {len(todo)} tiers with {n_workers} workers…")

    with Pool(n_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(write_shards_from_file, pack_tasks),
                total=len(pack_tasks),
                desc="Packing shards",
            )
        )

    # Aggregate per tier
    from collections import defaultdict

    agg: dict[str, dict] = defaultdict(lambda: {"written": 0, "skipped": 0, "shards": 0})
    for r in results:
        agg[r["tier_path"]]["written"] += r["written"]
        agg[r["tier_path"]]["skipped"] += r["skipped"]
        agg[r["tier_path"]]["shards"] += r["shards"]

    log.info("\nPacking complete:")
    grand_written = grand_skipped = grand_shards = 0
    for tp in sorted(agg):
        r = agg[tp]
        log.info(f"  {tp}: {r['written']:,} written, {r['skipped']:,} skipped, {r['shards']} shards")
        grand_written += r["written"]
        grand_skipped += r["skipped"]
        grand_shards += r["shards"]
    log.info(f"\nGRAND TOTAL: {grand_written:,} samples in {grand_shards} shards ({grand_skipped:,} skipped)")


if __name__ == "__main__":
    main()
