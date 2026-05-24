#!/usr/bin/env python3
"""
Build parquet mirror for: mulberry_1, mulberry_2, mapqa, zhwiki_1, zhwiki_2.

Steps:
  1. Extract archives if not already done (mulberry tar, mapqa zips)
  2. Build one parquet per dataset in parallel (5 processes)
  3. Skip rows with any missing image (no nulls)

Usage:
    build_parquet_other.py              # all 5 datasets
    build_parquet_other.py --dataset X  # single dataset
    build_parquet_other.py --test N     # first N rows per dataset
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ── Paths ─────────────────────────────────────────────────────────────────────
NV_BASE = Path(
    "/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/hf___nvidia___Nemotron-Image-Training-v3"
)
DST_BASE = Path("/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/swissai___Nemotron-Image-Training-v3")
DS_BASE = Path("/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/datasets")

MULBERRY_TAR = DS_BASE / "mulberry_1/hf___HuanjinYao___Mulberry-SFT/mulberry_images.tar"
MULBERRY_ROOT = DS_BASE / "mulberry_1/hf___HuanjinYao___Mulberry-SFT/mulberry_images"
MAPQA_DIR = DS_BASE / "mapqa"

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("messages", pa.string()),
        ("images", pa.map_(pa.string(), pa.binary())),
    ]
)
BATCH_ROWS = 64
READ_THREADS = 32


# ── Extraction ────────────────────────────────────────────────────────────────


def extract_mulberry():
    if MULBERRY_ROOT.exists():
        print("[setup] mulberry already extracted", flush=True)
        return
    print("[setup] extracting mulberry_images.tar (~21 GB)...", flush=True)
    subprocess.run(["tar", "-xf", str(MULBERRY_TAR), "-C", str(MULBERRY_TAR.parent)], check=True)
    print("[setup] mulberry extraction done", flush=True)


def extract_mapqa():
    for split in ["MapQA_S", "MapQA_U", "MapQA_V"]:
        if (MAPQA_DIR / split / "images").exists():
            print(f"[setup] {split} already extracted", flush=True)
            continue
        zip_path = MAPQA_DIR / f"{split}.zip"
        print(f"[setup] extracting {zip_path.name}...", flush=True)
        with zipfile.ZipFile(zip_path) as zf:
            members = [m for m in zf.namelist() if "/images/" in m and not m.endswith("/")]
            zf.extractall(MAPQA_DIR, members=members)
        print(f"[setup] {split} done", flush=True)


# ── Parquet builder ───────────────────────────────────────────────────────────


def build_parquet(task: dict) -> tuple:
    name = task["name"]
    jsonl_path = task["jsonl"]
    img_root = task.get("img_root")  # Path or None
    img_tar = task.get("img_tar")  # Path to tar (zhwiki)
    test_limit = task.get("test_limit", 0)

    dst_dir = DST_BASE / name
    dst = dst_dir / f"{name}.parquet"
    tmp = dst_dir / f"{name}.parquet.tmp"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    tmp.unlink(missing_ok=True)

    # For zhwiki: load entire shard tar into memory dict {filename: bytes}
    img_dict = None
    if img_tar:
        print(f"[{name}] loading {img_tar.name} into memory...", flush=True)
        img_dict = {}
        with tarfile.open(img_tar) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    f = tf.extractfile(member)
                    if f:
                        img_dict[Path(member.name).name] = f.read()
        print(f"[{name}] loaded {len(img_dict)} images", flush=True)

    def load_image(img_path: str) -> bytes | None:
        if img_dict is not None:
            return img_dict.get(Path(img_path).name)
        try:
            return (img_root / img_path).read_bytes()
        except OSError:
            return None

    t0 = time.monotonic()
    written = skipped = 0

    with (
        open(jsonl_path) as f,
        pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer,
        ThreadPoolExecutor(max_workers=READ_THREADS) as pool,
    ):
        batch_ids, batch_msgs, batch_imgs = [], [], []

        def flush():
            writer.write_table(
                pa.table(
                    {
                        "id": batch_ids,
                        "messages": batch_msgs,
                        "images": pa.array(batch_imgs, type=pa.map_(pa.string(), pa.binary())),
                    },
                    schema=SCHEMA,
                )
            )
            batch_ids.clear()
            batch_msgs.clear()
            batch_imgs.clear()

        for lineno, line in enumerate(f):
            if test_limit and lineno >= test_limit:
                break
            row = json.loads(line)
            img_paths = [
                item["image"]
                for msg in row["messages"]
                if msg["role"] == "user"
                for item in (msg["content"] if isinstance(msg["content"], list) else [])
                if isinstance(item, dict) and item.get("type") == "image"
            ]
            img_bytes = list(pool.map(load_image, img_paths))
            if any(b is None for b in img_bytes):
                skipped += 1
                continue

            batch_ids.append(row["id"])
            batch_msgs.append(json.dumps(row["messages"]))
            batch_imgs.append(list(zip(img_paths, img_bytes)))
            written += 1

            if len(batch_ids) >= BATCH_ROWS:
                flush()
            if written % 2000 == 0 and written > 0:
                elapsed = time.monotonic() - t0
                print(
                    f"[{name}] {written} rows, {skipped} skipped  {written / elapsed:.1f} rows/s",
                    flush=True,
                )

        if batch_ids:
            flush()

    os.rename(tmp, dst)
    elapsed = time.monotonic() - t0
    print(
        f"[{name}] DONE: {written} rows, {skipped} skipped  in {elapsed / 60:.1f} min",
        flush=True,
    )
    return name, written, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_DATASETS = ["mulberry_1", "mulberry_2", "mapqa", "zhwiki_1", "zhwiki_2"]


def make_tasks(names, test_limit):
    tasks = {
        "mulberry_1": {
            "name": "mulberry_1",
            "jsonl": NV_BASE / "mulberry_1/mulberry_1.jsonl",
            "img_root": MULBERRY_ROOT,
        },
        "mulberry_2": {
            "name": "mulberry_2",
            "jsonl": NV_BASE / "mulberry_2/mulberry_2.jsonl",
            "img_root": MULBERRY_ROOT,
        },
        "mapqa": {
            "name": "mapqa",
            "jsonl": NV_BASE / "mapqa/mapqa.jsonl",
            "img_root": MAPQA_DIR,
        },
        "zhwiki_1": {
            "name": "zhwiki_1",
            "jsonl": NV_BASE / "zhwiki_1/zhwiki_1.jsonl",
            "img_tar": NV_BASE / "zhwiki_1/media/shard_000000.tar",
        },
        "zhwiki_2": {
            "name": "zhwiki_2",
            "jsonl": NV_BASE / "zhwiki_2/zhwiki_2.jsonl",
            "img_tar": NV_BASE / "zhwiki_2/media/shard_000000.tar",
        },
    }
    return [{**tasks[n], "test_limit": test_limit} for n in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()

    names = [args.dataset] if args.dataset else ALL_DATASETS

    if any(n in names for n in ["mulberry_1", "mulberry_2"]):
        extract_mulberry()
    if "mapqa" in names:
        extract_mapqa()
    tasks = make_tasks(names, args.test)

    if len(tasks) == 1:
        build_parquet(tasks[0])
    else:
        with mp.Pool(processes=len(tasks)) as pool:
            results = pool.map(build_parquet, tasks)
        print("\n=== Summary ===")
        for name, written, skipped in results:
            print(f"  {name}: {written} rows, {skipped} skipped")


if __name__ == "__main__":
    main()
