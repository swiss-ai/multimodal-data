#!/usr/bin/env python3
"""
Build parquet mirror of long_document_arxiv_{1,2,3} with embedded images.

Skips rows where any referenced image is missing (version mismatch).
Writes to .tmp first then renames atomically (crash-safe).
Runs 3 splits in parallel.

Usage:
    build_parquet.py              # all 3 splits in parallel
    build_parquet.py --split N    # single split (1, 2, or 3)
    build_parquet.py --test N     # first N rows per split
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC_BASE = Path(
    "/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/hf___nvidia___Nemotron-Image-Training-v3"
)
DST_BASE = Path("/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/swissai___Nemotron-Image-Training-v3")
IMG_DIR = Path("/tmp/toolbox/download_arxiv/data/images")

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("messages", pa.string()),
        ("images", pa.map_(pa.string(), pa.binary())),
    ]
)

BATCH_ROWS = 64  # rows per parquet row group — keeps peak RAM bounded
READ_THREADS = 32  # threads for parallel image reads per batch


def load_image(path: Path):
    try:
        return path.read_bytes()
    except OSError:
        return None


def build_split(split: int, test_limit: int = 0):
    name = f"long_document_arxiv_{split}"
    src = SRC_BASE / name / f"{name}.jsonl"
    dst_dir = DST_BASE / name
    dst = dst_dir / f"{name}.parquet"
    tmp = dst_dir / f"{name}.parquet.tmp"

    dst_dir.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        print(f"[split {split}] already exists, removing to rebuild", flush=True)
        dst.unlink()
    tmp.unlink(missing_ok=True)

    t0 = time.monotonic()
    written = skipped = 0

    with (
        open(src) as f,
        pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer,
        ThreadPoolExecutor(max_workers=READ_THREADS) as pool,
    ):
        batch_ids = []
        batch_msgs = []
        batch_imgs = []

        def flush():
            table = pa.table(
                {
                    "id": batch_ids,
                    "messages": batch_msgs,
                    "images": pa.array(batch_imgs, type=pa.map_(pa.string(), pa.binary())),
                },
                schema=SCHEMA,
            )
            writer.write_table(table)
            batch_ids.clear()
            batch_msgs.clear()
            batch_imgs.clear()

        for lineno, line in enumerate(f):
            if test_limit and lineno >= test_limit:
                break

            row = json.loads(line)

            # Collect image paths referenced in this row
            img_paths = []
            for msg in row["messages"]:
                if msg["role"] != "user":
                    continue
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_paths.append(item["image"])

            # Load all images in parallel
            full_paths = [IMG_DIR / p for p in img_paths]
            img_bytes = list(pool.map(load_image, full_paths))

            # Skip row if any image is missing
            if any(b is None for b in img_bytes):
                skipped += 1
                continue

            batch_ids.append(row["id"])
            batch_msgs.append(json.dumps(row["messages"]))
            batch_imgs.append(list(zip(img_paths, img_bytes)))
            written += 1

            if len(batch_ids) >= BATCH_ROWS:
                flush()

            if written % 500 == 0 and written > 0:
                elapsed = time.monotonic() - t0
                print(
                    f"[split {split}] {written} rows written, {skipped} skipped  {written / elapsed:.1f} rows/s",
                    flush=True,
                )

        if batch_ids:
            flush()

    # Atomic rename
    os.rename(tmp, dst)
    elapsed = time.monotonic() - t0
    print(
        f"[split {split}] DONE: {written} rows, {skipped} skipped  in {elapsed / 60:.1f} min  →  {dst}",
        flush=True,
    )
    return written, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, choices=[1, 2, 3], default=0)
    parser.add_argument("--test", type=int, default=0, help="limit rows per split")
    args = parser.parse_args()

    splits = [args.split] if args.split else [1, 2, 3]

    if len(splits) == 1:
        build_split(splits[0], args.test)
    else:
        with mp.Pool(processes=3) as pool:
            results = pool.starmap(build_split, [(s, args.test) for s in splits])
        total_w = sum(r[0] for r in results)
        total_s = sum(r[1] for r in results)
        print(f"\nAll splits done: {total_w} rows written, {total_s} skipped total")


if __name__ == "__main__":
    main()
