#!/usr/bin/env python3
"""Repack Nemotron-Image-Training-v3 shards with pathologically large row groups.

Several Nemotron-v3 subsets ship parquet files with a single huge row group
(up to 54 GB in cc3m/train-00000.parquet). pyarrow's `iter_batches` can stream
such row groups in principle, but the reader still memory-maps the entire row
group's column chunks, causing host RAM OOMs in the downstream tokenization
loader (queue × 4-ranks-per-node × ~50 GB easily exceeds 856 GB node RAM).

This script identifies offending shards (any row group above `--threshold-gb`)
and rewrites them with bounded row groups (`--row-group-size` rows each),
streaming via `iter_batches` so we don't materialize the offending row group
ourselves. Output goes to a parallel directory; originals are not modified.

Typical use:

    python repack_nemotron_v3.py \\
        --src /capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/nemotron_image_training_v3/swissai___Nemotron-Image-Training-v3/ \\
        --dst /capstor/scratch/cscs/xyixuan/nemotron_repacked/ \\
        --threshold-gb 2.0 \\
        --row-group-size 2048 \\
        --workers 6

After repack, build a shadow directory (symlinks: offending shards → repacked
copies, others → originals) and point the tokenizer's manifest scan at it.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("repack_nemotron_v3")


def find_offending_shards(src_root: Path, threshold_bytes: int) -> list[dict]:
    """Return per-shard metadata for all shards with any row group >= threshold."""
    offending = []
    for subset_dir in sorted(src_root.iterdir()):
        if not subset_dir.is_dir():
            continue
        for shard in sorted(subset_dir.iterdir()):
            if not shard.name.endswith(".parquet"):
                continue
            try:
                pf = pq.ParquetFile(str(shard))
            except Exception:
                continue
            md = pf.metadata
            max_rg_bytes = max(
                md.row_group(i).total_byte_size for i in range(md.num_row_groups)
            )
            if max_rg_bytes >= threshold_bytes:
                offending.append({
                    "subset": subset_dir.name,
                    "shard": shard.name,
                    "src_path": str(shard),
                    "max_rg_gb": max_rg_bytes / 1e9,
                    "num_rg": md.num_row_groups,
                    "num_rows": md.num_rows,
                    "file_size_gb": os.path.getsize(shard) / 1e9,
                })
    offending.sort(key=lambda r: -r["max_rg_gb"])
    return offending


def _repack_one(args) -> dict:
    item, dst_root, row_group_size = args
    src = Path(item["src_path"])
    out_dir = Path(dst_root) / item["subset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / item["shard"]
    tmp_path = out_dir / (item["shard"] + ".tmp")
    expected_rows = item["num_rows"]

    if out_path.exists():
        try:
            existing = pq.ParquetFile(str(out_path)).metadata.num_rows
            if existing == expected_rows:
                return {"status": "skip-already-done", **item, "rows": existing}
        except Exception:
            pass  # corrupted, redo

    t0 = time.time()
    pf = pq.ParquetFile(str(src))
    schema = pf.schema_arrow

    rows_written = 0
    with pq.ParquetWriter(str(tmp_path), schema, compression="zstd") as writer:
        for batch in pf.iter_batches(batch_size=row_group_size):
            writer.write_batch(batch)
            rows_written += batch.num_rows

    if rows_written != expected_rows:
        tmp_path.unlink(missing_ok=True)
        return {
            "status": "row-mismatch",
            **item,
            "expected": expected_rows,
            "written": rows_written,
        }

    tmp_path.rename(out_path)
    elapsed = time.time() - t0

    new_md = pq.ParquetFile(str(out_path)).metadata
    return {
        "status": "ok",
        **item,
        "rows": rows_written,
        "elapsed_s": round(elapsed, 1),
        "new_num_row_groups": new_md.num_row_groups,
        "out_size_gb": round(os.path.getsize(out_path) / 1e9, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Root of the raw Nemotron dataset tree (contains <subset>/<shard>.parquet)",
    )
    p.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output root; repacked shards land under <dst>/<subset>/<shard>.parquet",
    )
    p.add_argument(
        "--threshold-gb",
        type=float,
        default=2.0,
        help="Shards with any row group >= threshold (GB) are repacked (default: 2.0)",
    )
    p.add_argument(
        "--row-group-size",
        type=int,
        default=2048,
        help="Target row count per row group in the output (default: 2048)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel repack workers (default: 6)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Identify offending shards and exit without writing",
    )
    args = p.parse_args()

    threshold_bytes = int(args.threshold_gb * 1e9)
    logger.info(f"Scanning {args.src} for row groups >= {args.threshold_gb} GB")
    offending = find_offending_shards(args.src, threshold_bytes)

    if not offending:
        logger.info("No offending shards. Nothing to do.")
        return

    logger.info(f"Found {len(offending)} offending shards:")
    for r in offending:
        logger.info(
            f"  {r['subset']:<28} {r['shard']:<35} "
            f"max_rg={r['max_rg_gb']:.2f}G rg={r['num_rg']} rows={r['num_rows']:,}"
        )

    if args.dry_run:
        logger.info("Dry-run; exiting without writing.")
        return

    args.dst.mkdir(parents=True, exist_ok=True)
    nproc = min(args.workers, len(offending))
    logger.info(f"Repacking with {nproc} workers → {args.dst}")

    t_start = time.time()
    results = []
    pool_args = [(item, args.dst, args.row_group_size) for item in offending]
    with mp.Pool(processes=nproc) as pool:
        for result in pool.imap_unordered(_repack_one, pool_args):
            results.append(result)
            tag = result["status"]
            extra = ""
            if tag == "ok":
                extra = (
                    f"rows={result['rows']:,} elapsed={result['elapsed_s']}s "
                    f"rg={result['new_num_row_groups']} size={result['out_size_gb']}G"
                )
            logger.info(
                f"  [{len(results):>2}/{len(offending)}] {tag:<18} "
                f"{result['subset']:<28} {result['shard']:<35} {extra}"
            )

    elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skip-already-done")
    n_err = len(results) - n_ok - n_skip
    logger.info(
        f"=== DONE in {elapsed:.0f}s ({elapsed/60:.1f} min) — "
        f"ok={n_ok} skipped={n_skip} errors={n_err} ==="
    )
    if n_err:
        for r in results:
            if r["status"] not in {"ok", "skip-already-done"}:
                logger.error(f"  {r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
