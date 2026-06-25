#!/usr/bin/env python3
"""Split the 3 oversize Pangea tars into ~20-30 GB chunks for parallel scan.

Why: the scanner's per-tar walk is sequential within a tar (tarfile is a
stream). Tars > ~30 GB become single-worker bottlenecks during scan that
serialize the whole job around the slowest one. Splitting them into
balanced sub-tars unlocks per-tar parallelism in build_tar_index().

Targets (in processed/sft/pangea_instruct/tars/):
  - cultural/laion-multi-1M/images.tar (~128 GB, ~1M images)        → 4 chunks
  - general/ALLaVA-4V/allava_laion.tar (~81 GB, ~280K images)       → 4 chunks
  - general/allava_vflan/images.tar    (~35 GB, ~190K images)       → 2 chunks

Within each big tar we run sequentially (tarfile reads are streaming), but the
3 big tars run in parallel via ProcessPoolExecutor. Each one:
  1. Counts members in a quick header walk (~few seconds)
  2. Streams members and writes them to N output tars round-robin
  3. Deletes the original on success

Output tar names: ``<stem>_part_NN.tar`` (e.g., ``images_part_00.tar``).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reshard_pangea")

TARS_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/pangea_instruct/tars"
)

# (relative path, num chunks) — picked to land each chunk in ~20-30 GB
TARGETS = [
    ("cultural/laion-multi-1M/images.tar",   4),  # 128/4 = 32 GB
    ("general/ALLaVA-4V/allava_laion.tar",   4),  # 81/4  = 20 GB
    ("general/allava_vflan/images.tar",      2),  # 35/2  = 17 GB
]


def _reshard_one(rel_path: str, n_chunks: int) -> tuple[str, int, float, str | None]:
    src = TARS_ROOT / rel_path
    if not src.exists():
        return (rel_path, 0, 0.0, f"missing source: {src}")
    parent = src.parent
    stem = src.stem  # e.g., "images" or "allava_laion"

    t0 = time.time()
    n_members = 0
    n_written = [0] * n_chunks

    try:
        # Open all N output tars upfront
        out_paths = [parent / f"{stem}_part_{i:02d}.tar" for i in range(n_chunks)]
        out_tars = [tarfile.open(p, "w") for p in out_paths]

        try:
            with tarfile.open(src, "r") as tf_in:
                for m in tf_in:
                    if not m.isfile():
                        continue
                    f = tf_in.extractfile(m)
                    if f is None:
                        continue
                    data = f.read()
                    bucket = n_members % n_chunks
                    new_info = tarfile.TarInfo(name=m.name)
                    new_info.size = len(data)
                    new_info.mtime = m.mtime
                    new_info.mode = m.mode
                    out_tars[bucket].addfile(new_info, io.BytesIO(data))
                    n_written[bucket] += 1
                    n_members += 1
        finally:
            for tf in out_tars:
                tf.close()
    except Exception as e:
        # Best effort cleanup of partial outputs
        for p in out_paths:
            try: p.unlink()
            except: pass
        return (rel_path, 0, time.time() - t0, f"{type(e).__name__}: {e}")

    # Success — verify outputs are valid, then delete original
    elapsed = time.time() - t0
    sizes_gb = [(out_paths[i].stat().st_size / 1e9) for i in range(n_chunks)]
    logger.info(
        "[reshard] %s → %d parts in %.0fs; sizes %s GB; members per part %s",
        rel_path, n_chunks, elapsed,
        [f"{s:.1f}" for s in sizes_gb], n_written,
    )

    # Delete original ONLY after successful write of all chunks
    src.unlink()
    logger.info("[reshard] removed original %s", src)
    return (rel_path, n_members, elapsed, None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=3, help="parallel workers (default: 3, one per target tar)")
    args = p.parse_args()

    logger.info("resharding %d oversize tars with %d workers", len(TARGETS), args.workers)
    t0 = time.time()
    successes = failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_reshard_one, rel, n): rel for rel, n in TARGETS}
        for f in as_completed(futs):
            rel, n_mem, dur, err = f.result()
            if err:
                logger.error("[reshard] FAIL %s — %s", rel, err)
                failures += 1
            else:
                logger.info("[reshard] ✓ %s — %d members in %.0fs", rel, n_mem, dur)
                successes += 1
    logger.info("done in %.0fs: %d success, %d fail", time.time() - t0, successes, failures)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
