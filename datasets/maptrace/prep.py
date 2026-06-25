#!/usr/bin/env python3
"""Reshape google/MapTrace from caption-style to SFT-style parquet.

Upstream MapTrace ships 4 columns per record:
  - image_bytes  (binary)   — the map image
  - input_text   (string)   — templated nav prompt with start/end coords
  - label_text   (string)   — list-of-tuples coord path (the supervision target)
  - map_description (string) — generic caption (not used in the SFT task)

Upstream layout: snapshots/<rev>/{floormaps, maptrace, maptrace_20k}/*.parquet
  - floormaps + maptrace = train (~2.18M rows total)
  - maptrace_20k = eval split used in the paper (DROPPED here — train-only rule)

Output: processed/sft/google_maptrace/{maptrace,floormaps}/shard_*.parquet with:
  - id            : string (hash of input)
  - image         : struct<bytes: binary, path: string>  (standard SFT image shape)
  - conversations : list<struct<role,content>>
      [ {role:'user', content:'<image>\\n' + input_text},
        {role:'assistant', content: label_text} ]

Pre-pending '<image>' to the user turn matches the EMUSft renderer's expected
contract (one placeholder per image; we have 1 image per record here).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prep_maptrace")

SRC_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache/datasets--google--MapTrace/snapshots"
)
DST_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/google_maptrace"
)
TRAIN_SPLITS = ("maptrace", "floormaps")
EVAL_SPLITS = ("maptrace_20k",)  # dropped — eval split used in the paper

# Output schema matches LLaVA-OV-style SFT
OUT_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("image", pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])),
    pa.field("conversations", pa.list_(pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ]))),
])


def _reshape_one(src_path: str) -> tuple[str, int, float, str | None]:
    rel = Path(src_path).relative_to(SRC_ROOT.glob("*").__next__())
    dst_path = DST_ROOT / rel
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        t = pq.read_table(src_path, columns=["image_bytes", "input_text", "label_text"])
        n = t.num_rows
        if n == 0:
            return (src_path, 0, time.time() - t0, None)

        image_bytes = t.column("image_bytes").to_pylist()
        input_text = t.column("input_text").to_pylist()
        label_text = t.column("label_text").to_pylist()

        ids = [hashlib.md5((str(it) + str(lt))[:512].encode()).hexdigest()[:16] for it, lt in zip(input_text, label_text)]
        images = [{"bytes": b, "path": None} for b in image_bytes]
        convs = [
            [
                {"role": "user", "content": "<image>\n" + (it or "")},
                {"role": "assistant", "content": lt or ""},
            ]
            for it, lt in zip(input_text, label_text)
        ]

        out = pa.table({
            "id": ids,
            "image": images,
            "conversations": convs,
        }, schema=OUT_SCHEMA)
        pq.write_table(out, dst_path, compression="zstd")
        return (src_path, n, time.time() - t0, None)
    except Exception as e:
        return (src_path, 0, time.time() - t0, f"{type(e).__name__}: {e}")


def _discover_train_shards() -> list[str]:
    snap_dir = next(SRC_ROOT.glob("*"))
    paths: list[str] = []
    for split in TRAIN_SPLITS:
        paths.extend(str(p) for p in sorted((snap_dir / split).glob("*.parquet")))
    logger.info("found %d train shards across splits %s", len(paths), TRAIN_SPLITS)
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=64)
    args = p.parse_args()

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    shards = _discover_train_shards()
    t0 = time.time()
    rows_done = 0
    successes = failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_reshape_one, p): p for p in shards}
        for i, f in enumerate(as_completed(futs)):
            src, n, dur, err = f.result()
            if err:
                logger.error("FAIL %s — %s", Path(src).name, err)
                failures += 1
            else:
                successes += 1
                rows_done += n
            if (i + 1) % 200 == 0:
                logger.info("[%4d/%d] rows=%d  elapsed=%.0fs", i + 1, len(shards), rows_done, time.time() - t0)
    logger.info("done in %.1fs: %d success, %d fail, %d rows", time.time() - t0, successes, failures, rows_done)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
