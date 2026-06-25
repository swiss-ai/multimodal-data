#!/usr/bin/env python3
"""Prep PangeaInstruct for the jsonl_tar tokenize pipeline.

Two steps, both safe to re-run (idempotent overwrite):

  1. Stream PangeaIns.json (a 12 GB JSON array) into PangeaIns_train.jsonl,
     dropping records whose ``image`` field starts with ``val2017/`` (the only
     val-split marker in the corpus — see audit on 2026-05-16).

  2. Re-tar the 10 train-split .zip image archives into uncompressed .tar files
     written alongside the originals. ``tarfile.open(..., "r")`` reads .tar +
     .tar.gz natively, but cannot read .zip, so the jsonl_tar scanner needs
     these in tar form. val2017.zip is skipped.

The conversions stream zip → tar in-process (no extraction to disk), so peak
memory per worker is bounded by the largest member, not the whole archive.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prep_pangea")

RAW_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___neulab___PangeaInstruct"
)
PROC_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/pangea_instruct"
)
PROC_TARS = PROC_ROOT / "tars"
JSON_IN = RAW_ROOT / "PangeaIns.json"
JSONL_OUT = PROC_ROOT / "PangeaIns_train.jsonl"

# 10 train-split zips to re-tar (val2017.zip explicitly excluded).
ZIPS_TO_RETAR = [
    "general/COCO/train2017.zip",
    "doc+chart/table-vqa/images.zip",
    "doc+chart/doc-vqa/images.zip",
    "doc+chart/Viet-Doc-VQA/images.zip",
    "doc+chart/Viet-Doc-VQA-II/images.zip",
    "general/llava-med-zh-instruct-60k/images.zip",
    "general/MTVQA/images.zip",
    "general/ShareGPT-4o/images.zip",
    "general/nlvr2-llava/images.zip",
    "general/allava_vflan/images.zip",
]
EXCLUDED_IMAGE_PREFIXES = ("val2017/",)


def convert_json_to_jsonl() -> None:
    """Stream-filter JSON array → JSONL, dropping val2017 image refs."""
    import ijson  # streaming JSON parser
    from decimal import Decimal

    # ijson decodes numeric literals as Decimal for arbitrary-precision; cast
    # back to native float/int when serializing (Pangea numbers are all in the
    # standard float64 range — no precision loss).
    def _json_default(o):
        if isinstance(o, Decimal):
            i = int(o)
            return i if i == o else float(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    JSONL_OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_kept = n_drop_val = 0

    logger.info("[jsonl] %s  →  %s", JSON_IN, JSONL_OUT)
    with open(JSON_IN, "rb") as src, open(JSONL_OUT, "w") as dst:
        for rec in ijson.items(src, "item"):
            img = rec.get("image")
            if isinstance(img, str) and any(img.startswith(p) for p in EXCLUDED_IMAGE_PREFIXES):
                n_drop_val += 1
                continue
            dst.write(json.dumps(rec, ensure_ascii=False, default=_json_default))
            dst.write("\n")
            n_kept += 1
            if (n_kept + n_drop_val) % 500_000 == 0:
                logger.info(
                    "[jsonl] processed %d records (kept %d, dropped %d val)",
                    n_kept + n_drop_val, n_kept, n_drop_val,
                )

    elapsed = time.time() - t0
    logger.info(
        "[jsonl] done in %.1fs: kept %d, dropped %d (val2017)",
        elapsed, n_kept, n_drop_val,
    )


def _retar_one(rel_zip_path: str) -> tuple[str, int, int, float, str | None]:
    """Stream a single zip → uncompressed tar under processed/sft/pangea_instruct/tars/."""
    zip_path = RAW_ROOT / rel_zip_path
    tar_path = (PROC_TARS / rel_zip_path).with_suffix(".tar")
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    try:
        n_members = 0
        with zipfile.ZipFile(zip_path) as zf, tarfile.open(tar_path, "w") as tf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                data = zf.read(info.filename)
                tarinfo = tarfile.TarInfo(name=info.filename)
                tarinfo.size = len(data)
                tarinfo.mtime = int(info.date_time and time.mktime(info.date_time + (0, 0, -1)) or t0)
                tarinfo.mode = 0o644
                tf.addfile(tarinfo, io.BytesIO(data))
                n_members += 1
    except Exception as e:
        return (rel_zip_path, 0, 0, time.time() - t0, f"{type(e).__name__}: {e}")

    return (rel_zip_path, n_members, tar_path.stat().st_size, time.time() - t0, None)


def retar_zips(workers: int) -> None:
    """Re-tar all 10 train zips in parallel."""
    logger.info("[retar] re-taring %d zips with %d workers", len(ZIPS_TO_RETAR), workers)
    t0 = time.time()
    successes = failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_retar_one, p): p for p in ZIPS_TO_RETAR}
        for f in as_completed(futs):
            rel, n_mem, sz, dur, err = f.result()
            if err:
                logger.error("[retar] FAIL %s — %s", rel, err)
                failures += 1
            else:
                logger.info(
                    "[retar] %s — %d members, %.1f GB out in %.1fs",
                    rel, n_mem, sz / 1e9, dur,
                )
                successes += 1
    logger.info(
        "[retar] done in %.1fs: %d success, %d fail",
        time.time() - t0, successes, failures,
    )
    if failures:
        sys.exit(1)


def _cat_one(target: str, parts: list[str]) -> tuple[str, int, float, str | None]:
    """Concatenate split tar parts → single tar under processed/sft/pangea_instruct/tars/."""
    target_path = PROC_TARS / target
    target_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        with open(target_path, "wb") as out:
            for p in parts:
                src = RAW_ROOT / p
                with open(src, "rb") as f:
                    while True:
                        chunk = f.read(64 * 1024 * 1024)  # 64 MB
                        if not chunk:
                            break
                        out.write(chunk)
    except Exception as e:
        return (target, 0, time.time() - t0, f"{type(e).__name__}: {e}")
    return (target, target_path.stat().st_size, time.time() - t0, None)


# Split-part tar groups per upstream Pangea HF README convention:
#   cat part_* > images.tar  (or .partaa/.partab for ALLaVA-4V)
SPLIT_PART_GROUPS = [
    {
        "target": "general/ALLaVA-4V/allava_laion.tar",
        "parts": [
            "general/ALLaVA-4V/allava_laion.tar.partaa",
            "general/ALLaVA-4V/allava_laion.tar.partab",
        ],
    },
    {
        "target": "cultural/laion-multi-1M/images.tar",
        "parts": [
            "cultural/laion-multi-1M/part_00",
            "cultural/laion-multi-1M/part_01",
            "cultural/laion-multi-1M/part_02",
            "cultural/laion-multi-1M/part_03",
        ],
    },
]


def cat_split_parts(workers: int) -> None:
    """Reassemble Pangea split-part tars per upstream HF README convention."""
    logger.info("[cat] reassembling %d split-tar groups", len(SPLIT_PART_GROUPS))
    t0 = time.time()
    successes = failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_cat_one, g["target"], g["parts"]): g["target"] for g in SPLIT_PART_GROUPS}
        for f in as_completed(futs):
            target, sz, dur, err = f.result()
            if err:
                logger.error("[cat] FAIL %s — %s", target, err)
                failures += 1
            else:
                logger.info("[cat] %s — %.1f GB in %.1fs", target, sz / 1e9, dur)
                successes += 1
    logger.info("[cat] done in %.1fs: %d success, %d fail", time.time() - t0, successes, failures)
    if failures:
        sys.exit(1)


def decontaminate(
    src_jsonl: Path = PROC_ROOT / "PangeaIns_train_normalized.jsonl",
    dst_jsonl: Path = PROC_ROOT / "PangeaIns_train_decontaminated.jsonl",
    ids_path: Path = Path(__file__).parent / "contaminated_ids.txt",
) -> None:
    """Filter normalized jsonl by record-id contamination set.

    Pangea ships 9,114 contamination IDs (mixed numeric + string) against
    multimodal benchmarks. The IDs match the ``id`` field of each record.
    Numeric IDs end in ``.0`` in the source file — we coerce both sides to
    a canonical string for membership testing.
    """
    t0 = time.time()
    raw = ids_path.read_text().strip()
    ids = {x.strip() for x in raw.split(",") if x.strip()}
    # Also accept the integer form of float-style IDs (e.g. "1000407002424.0" → "1000407002424")
    extra: set[str] = set()
    for x in ids:
        if x.endswith(".0"):
            extra.add(x[:-2])
        try:
            extra.add(str(int(float(x))))
        except (ValueError, OverflowError):
            pass
    ids |= extra
    logger.info("[decontaminate] loaded %d contamination IDs (incl. coerced forms)", len(ids))

    kept = dropped = 0
    with open(src_jsonl) as src, open(dst_jsonl, "w") as dst:
        for line in src:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("id", ""))
            if rid in ids:
                dropped += 1
                continue
            dst.write(line)
            kept += 1
    logger.info(
        "[decontaminate] done in %.1fs: kept %d, dropped %d (%.4f%%)",
        time.time() - t0, kept, dropped, 100 * dropped / max(1, kept + dropped),
    )
    logger.info("[decontaminate] output: %s", dst_jsonl)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", default="jsonl,retar",
                   help="comma-separated subset of {jsonl,retar,cat_parts,decontaminate}")
    p.add_argument("--retar-workers", type=int, default=10)
    p.add_argument("--cat-workers", type=int, default=2,
                   help="parallel workers for cat (default: 2, one per split-group)")
    args = p.parse_args()
    steps = set(s.strip() for s in args.steps.split(","))

    if "jsonl" in steps:
        convert_json_to_jsonl()
    if "retar" in steps:
        retar_zips(args.retar_workers)
    if "cat_parts" in steps:
        cat_split_parts(args.cat_workers)
    if "decontaminate" in steps:
        decontaminate()


if __name__ == "__main__":
    main()
