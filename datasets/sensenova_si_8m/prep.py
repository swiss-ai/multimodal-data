#!/usr/bin/env python3
"""Prep SenseNova-SI-8M for the jsonl_tar tokenize pipeline.

SenseNova ships with:
  - SenseNova-SI-8M.jsonl  (8.16M-row master manifest, ready as-is)
  - images_part_001..052.zip  (52 STORE-mode image zips, ~20 GB each)

This script re-tars the 52 zips into 52 uncompressed .tar files written
alongside the originals. tarfile.open(..., "r") reads .tar natively; it can't
read .zip, so the jsonl_tar scanner needs these in tar form.

Because the zips are STORE-mode (no compression), the conversion is
metadata-only repackaging — fast, no decompression CPU. Streams zip→tar
in-process; peak memory per worker bounded by the largest member.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prep_sensenova")

RAW_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___sensenova___SenseNova-SI-8M"
)
PROC_TARS = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/sensenova_si_8m"
)


def _list_zips() -> list[str]:
    paths = sorted(RAW_ROOT.glob("images_part_*.zip"))
    return [str(p.relative_to(RAW_ROOT)) for p in paths]


def _retar_one(rel_zip_path: str) -> tuple[str, int, int, float, str | None]:
    """Stream a single zip → uncompressed tar under processed/sft/sensenova_si_8m/tars/."""
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
                tarinfo.mtime = int(t0)
                tarinfo.mode = 0o644
                tf.addfile(tarinfo, io.BytesIO(data))
                n_members += 1
    except Exception as e:
        return (rel_zip_path, 0, 0, time.time() - t0, f"{type(e).__name__}: {e}")

    return (rel_zip_path, n_members, tar_path.stat().st_size, time.time() - t0, None)


def retar_zips(workers: int) -> None:
    """Re-tar all 52 image zips in parallel."""
    zips = _list_zips()
    logger.info("[retar] re-taring %d zips with %d workers", len(zips), workers)
    if not zips:
        logger.error("no zips found under %s", RAW_ROOT)
        sys.exit(1)
    t0 = time.time()
    successes = failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_retar_one, p): p for p in zips}
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--retar-workers", type=int, default=32,
                   help="parallel workers for re-tar (default: 32; 52 zips fit in 2 waves)")
    args = p.parse_args()
    retar_zips(args.retar_workers)


if __name__ == "__main__":
    main()
