"""Filter a FineVision subset by quality ratings.

Drops rows where `visual_dependency_min < 2` (Q doesn't actually need the image)
OR `image_correspondence_min < 2` (answer factually wrong given image).
These are the two "factual" dimensions — filtering them removes the obvious-
garbage tier (~3-10% of any given subset) while preserving the breadth that
the FineVision paper's ablation favors.

`relevance_min` and `formatting_min` are NOT filtered (stylistic noise; paper
found filtering them hurts downstream performance).

Output schema is the input minus the 8 rating columns — only `images`,
`texts`, `source` survive (and that's all the tokenize parser needs).

Run as standalone per subset:
    python filter_finevision.py SynthChartNet
"""

from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc


IN_ROOT = "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision"
OUT_ROOT = "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision"

VD_MIN = 2  # visual_dependency_min threshold (>=)
IC_MIN = 2  # image_correspondence_min threshold (>=)


def filter_subset(subset: str) -> dict:
    """Filter one subset's parquet shards. Returns aggregate stats."""
    in_dir = Path(IN_ROOT) / subset
    out_dir = Path(OUT_ROOT) / subset
    if not in_dir.is_dir():
        raise FileNotFoundError(f"{in_dir} not found")

    files = sorted(in_dir.glob("train-*.parquet"))
    if not files:
        raise RuntimeError(f"no train-*.parquet in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    keep_cols = ["images", "texts", "source"]
    filt_cols = ["visual_dependency_min", "image_correspondence_min"]

    total_in = 0
    total_out = 0
    for f in files:
        out_path = out_dir / f.name
        if out_path.exists():
            tbl = pq.read_table(str(out_path))
            total_out += tbl.num_rows
            print(f"  {f.name}: SKIP (already exists, {tbl.num_rows:,} rows)", flush=True)
            # Need n_in for stats — read just the count from input
            n_in = pq.ParquetFile(str(f)).metadata.num_rows
            total_in += n_in
            continue

        tbl = pq.read_table(str(f), columns=keep_cols + filt_cols)
        n_in = tbl.num_rows
        mask = pc.and_(
            pc.greater_equal(tbl["visual_dependency_min"], VD_MIN),
            pc.greater_equal(tbl["image_correspondence_min"], IC_MIN),
        )
        filtered = tbl.filter(mask).select(keep_cols)
        n_out = filtered.num_rows
        pq.write_table(filtered, str(out_path), compression="zstd")
        total_in += n_in
        total_out += n_out
        pct = 100.0 * n_out / n_in if n_in else 0.0
        print(f"  {f.name}: {n_in:,} -> {n_out:,} ({pct:.1f}% kept)", flush=True)

    pct = 100.0 * total_out / total_in if total_in else 0.0
    print(f"=== {subset}: {total_in:,} -> {total_out:,} ({pct:.1f}% kept) ===", flush=True)
    return {"subset": subset, "in": total_in, "out": total_out, "pct": pct}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subset", help="FineVision subset name (e.g. SynthChartNet)")
    args = p.parse_args()
    filter_subset(args.subset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
