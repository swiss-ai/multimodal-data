#!/usr/bin/env python3
"""Prep infly/Infinity-Doc2-5M for the jsonl_tar tokenize pipeline.

Each of the 10 source ``images_labels.tar.gz`` archives bundles BOTH a
``labels/labels.jsonl`` annotation file AND an ``images/`` dir of payloads in
one gzip-compressed tar. This script streams each archive, separating the two:

  - ``labels/labels.jsonl`` → written as a sibling ``labels.jsonl`` (uncompressed)
  - ``images/...`` members  → re-packed into an uncompressed ``images.tar``

Output layout (parent_dir scope for jsonl_tar scanner):

  processed/sft/infinity_doc2_5m/
    <task>/<subtask>/
      labels.jsonl     # image refs unchanged: ``images/part0001/<hash>.jpg``
      images.tar       # member names match the refs

Streams tarfile → tarfile (no fs extraction); decompresses gzip once, writes
uncompressed (~same I/O cost as gunzip + extract, but no per-file metadata ops
on /capstor).
"""

from __future__ import annotations

import argparse
import logging
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prep_infinity")

RAW_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___infly___Infinity-Doc2-5M"
)
PROC_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/infinity_doc2_5m"
)

# 10 source archives (task / subtask paths under RAW_ROOT).
ARCHIVES: list[tuple[str, str]] = [
    ("Blank_Page_Parsing",      "mix"),
    ("Chart_Parsing",           "chart2table"),
    ("Chart_Parsing",           "chart2code"),
    ("Chart_Parsing",           "chart2json"),
    ("Chemical_Formula_Parsing","chem2smiles"),
    ("Document_VQA",            "docvqa"),
    ("Element_Parsing",         "table2html"),
    ("Element_Parsing",         "formula2latex"),
    ("Element_Parsing",         "table2md"),
    ("Layout_Analysis",         "layout_analysis"),
]


def _split_one(task: str, subtask: str) -> tuple[str, dict, str | None]:
    """Stream one images_labels.tar.gz → (labels.jsonl, images.tar)."""
    src = RAW_ROOT / task / subtask / "images_labels.tar.gz"
    out_dir = PROC_ROOT / task / subtask
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_out = out_dir / "labels.jsonl"
    tar_out = out_dir / "images.tar"
    t0 = time.time()
    stats = {"n_image_members": 0, "n_labels_bytes": 0, "elapsed_s": 0.0}

    try:
        with tarfile.open(src, "r") as src_tf, \
             tarfile.open(tar_out, "w") as dst_tf, \
             open(labels_out, "wb") as labels_f:
            for m in src_tf:
                if not m.isfile():
                    continue
                if m.name.startswith("labels/"):
                    # Stream labels content out to flat file
                    f = src_tf.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                    labels_f.write(data)
                    stats["n_labels_bytes"] += len(data)
                elif m.name.startswith("images/"):
                    # Copy image member into dst tar (preserve original member name)
                    f = src_tf.extractfile(m)
                    if f is None:
                        continue
                    dst_tf.addfile(m, f)
                    stats["n_image_members"] += 1
    except Exception as e:
        return (f"{task}/{subtask}", stats, f"{type(e).__name__}: {e}")

    stats["elapsed_s"] = time.time() - t0
    return (f"{task}/{subtask}", stats, None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=10,
                   help="parallel workers (default: 10, one per archive)")
    args = p.parse_args()

    logger.info("Splitting %d archives with %d workers", len(ARCHIVES), args.workers)
    t0 = time.time()
    successes = failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_split_one, t, s): (t, s) for t, s in ARCHIVES}
        for f in as_completed(futs):
            name, st, err = f.result()
            if err:
                logger.error("FAIL %s — %s", name, err)
                failures += 1
            else:
                logger.info(
                    "✓ %-50s %7d images, %5.1f MB labels, %.1fs",
                    name, st["n_image_members"], st["n_labels_bytes"] / 1e6, st["elapsed_s"],
                )
                successes += 1
    logger.info("Done in %.1fs: %d success, %d fail", time.time() - t0, successes, failures)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
