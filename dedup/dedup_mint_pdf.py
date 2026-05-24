import argparse
import glob
import io
import os
from multiprocessing import Pool

from PIL import Image, ImageSequence

# fmt:off
# Remove this line - set HF_TOKEN env var before running
os.environ.setdefault("HF_HOME", os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface")))
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
# fmt:on

import cv2
import numpy as np
import pandas as pd
import webdataset as wds
from rocksdict import Options, Rdict, WriteBatch

from datasets import Image as HFImage
from datasets import load_dataset

# ==========================================
# CONFIGURATION
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--num_cpus", type=int, required=True)
args = parser.parse_args()

HF_DATASET_NAME = args.dataset
HF_SPLIT = "train"

RESULT_DIR = "/tmp/toolbox/deduplicate_stage_1/results"
DB_PATH = os.path.join(RESULT_DIR, "db_arxiv")
HASH_DIR = os.path.join(RESULT_DIR, HF_DATASET_NAME)
REJECT_LIST = os.path.join(HASH_DIR, "reject_list.txt")

ROOT_OUTPUT_DIR = "/path/to/data/vision-datasets"
OUTPUT_DIR = os.path.join(ROOT_OUTPUT_DIR, HF_DATASET_NAME)

# Resources
NUM_PROCESSES = args.num_cpus
ROCKSDB_BG_JOBS = 4

# Global variable for Stage 3 workers
reject_set = None

# ==========================================
# STAGE 1: HASHING
# ==========================================


def compute_phash(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    resized = cv2.resize(gray, (32, 32))
    phash = cv2.img_hash.pHash(resized)
    return phash.tobytes().hex()


def process_shard_hashing(worker_id, total_workers):
    print(f"[Stage 1] Worker {worker_id} starting stream...")

    cache_dir = os.environ.get("HF_HUB_CACHE")
    d_org, d_name = HF_DATASET_NAME.split("/", 1)
    dataset_path = f"{cache_dir}/datasets--{d_org}--{d_name}/snapshots/**/*.tar"
    local_files = sorted(glob.glob(dataset_path, recursive=True))
    if not local_files:
        raise FileNotFoundError(f"Could not find any parquet files at {dataset_path}")
    worker_files = [f for i, f in enumerate(local_files) if i % total_workers == worker_id]

    ds = (
        load_dataset(
            "webdataset",
            data_files=worker_files,
            split="train",
            streaming=True,
        )
        .select_columns(["__key__", "tiff"])
        .cast_column("tiff", HFImage(decode=False))
    )

    # ds = ds.take(10)

    records = []
    chunk_idx = 0

    for item in ds:
        img_data = item["tiff"]
        img_bytes = img_data["bytes"]
        key = item["__key__"]

        with Image.open(io.BytesIO(img_bytes)) as img:
            for i, page in enumerate(ImageSequence.Iterator(img)):
                page_key = f"{key}_{i:03d}"

                width, height = page.size
                if not (200 <= width <= 8000 and 200 <= height <= 8000):
                    continue

                page = page.convert("RGB")
                nparr = np.array(page)
                h = compute_phash(nparr)
                records.append({"key": page_key, "hash": h})

            if len(records) >= 10000:
                fn = f"hashes_{worker_id:03}_{chunk_idx:04d}.parquet"
                output_path = os.path.join(HASH_DIR, fn)
                pd.DataFrame(records).to_parquet(output_path)
                chunk_idx += 1
                records = []

    if records:
        fn = f"hashes_{worker_id:03}_{chunk_idx:04d}_end.parquet"
        output_path = os.path.join(HASH_DIR, fn)
        pd.DataFrame(records).to_parquet(output_path)

    print(f"[Stage 1] Worker {worker_id} finished processing {chunk_idx + 1} chunks.")
    return True


# ==========================================
# STAGE 2: DEDUPLICATION
# ==========================================


def run_stage_2_deduplication():
    print("--- Starting Stage 2: Database Check & Reject List Gen ---")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    opts = Options()
    opts.create_if_missing(True)
    opts.set_max_background_jobs(ROCKSDB_BG_JOBS)
    opts.set_write_buffer_size(64 * 1024 * 1024)  # 64MB

    db = Rdict(DB_PATH, options=opts)

    hash_files = sorted(glob.glob(f"{HASH_DIR}/*.parquet"))
    duplicates = []
    total_new = 0
    total_dupes = 0

    print(f"Scanning {len(hash_files)} hash files against persistent DB...")

    for hi, f in enumerate(hash_files):
        print(f"Processing file {hi + 1}/{len(hash_files)}: {os.path.basename(f)}")

        df = pd.read_parquet(f)
        wb = WriteBatch()
        seen_in_batch = set()
        batch_writes = 0

        keys = df["key"].values
        hashes = df["hash"].values

        for key, img_hash in zip(keys, hashes):
            if img_hash in db or img_hash in seen_in_batch:
                duplicates.append(key)
                total_dupes += 1
            else:
                wb[img_hash] = key.encode("utf-8")
                seen_in_batch.add(img_hash)
                batch_writes += 1
                total_new += 1

        if batch_writes > 0:
            db.write(wb)

    print("Finished scanning. Closing DB...")
    db.close()

    print(f"Saving {len(duplicates)} duplicates to {REJECT_LIST}...")
    with open(REJECT_LIST, "w") as f:
        for key in duplicates:
            f.write(f"{key}\n")

    print(f"Stage 2 Done. New items: {total_new}, Duplicates found: {total_dupes}")
    return True


# ==========================================
# STAGE 3: REWRITING
# ==========================================


def load_reject_list(path):
    s = set()
    with open(path, "r") as f:
        for line in f:
            s.add(line.strip())
    return s


def init_worker_rewrite(reject_path):
    global reject_set
    reject_set = load_reject_list(reject_path)


def process_shard_rewrite(worker_id, total_workers):
    global reject_set
    assert reject_set is not None

    output_filename = f"shard_{worker_id:03d}.tar"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if os.path.exists(output_path):
        print(f"[Stage 3] Output exists, skipping: {output_filename}")
        return

    print(f"[Stage 3] Worker {worker_id} starting rewrite to {output_filename}...")

    cache_dir = os.environ.get("HF_HUB_CACHE")
    d_org, d_name = HF_DATASET_NAME.split("/", 1)
    dataset_path = f"{cache_dir}/datasets--{d_org}--{d_name}/snapshots/**/*.tar"
    local_files = sorted(glob.glob(dataset_path, recursive=True))
    if not local_files:
        raise FileNotFoundError(f"Could not find any parquet files at {dataset_path}")
    worker_files = [f for i, f in enumerate(local_files) if i % total_workers == worker_id]

    ds = (
        load_dataset(
            "webdataset",
            data_files=worker_files,
            split="train",
            streaming=True,
        )
        .select_columns(["__key__", "tiff"])
        .cast_column("tiff", HFImage(decode=False))
    )

    # ds = ds.take(10)

    sink = wds.TarWriter(output_path)  # type:ignore
    kept = 0
    dropped = 0

    for item in ds:
        img_data = item["tiff"]
        img_bytes = img_data["bytes"]
        key = item["__key__"]

        with Image.open(io.BytesIO(img_bytes)) as img:
            for i, page in enumerate(ImageSequence.Iterator(img)):
                page_key = f"{key}_{i:03d}"
                if page_key in reject_set:
                    dropped += 1
                    continue

                width, height = page.size
                if not (200 <= width <= 8000 and 200 <= height <= 8000):
                    dropped += 1
                    continue

                page = page.convert("RGB")
                out = io.BytesIO()
                page.save(out, format="JPEG", quality=95)

                sink.write({"__key__": page_key, "jpg": out.getvalue()})
                kept += 1

    sink.close()
    return f"Worker {worker_id}: Kept {kept}, Dropped {dropped}"


# ==========================================
# MAIN
# ==========================================


def run_stage_1_hashing():
    print("--- Starting Stage 1: Hashing ---")
    os.makedirs(HASH_DIR, exist_ok=True)

    args = [(i, NUM_PROCESSES) for i in range(NUM_PROCESSES)]
    with Pool(NUM_PROCESSES) as pool:
        pool.starmap(process_shard_hashing, args)

    return True


def run_stage_3_rewriting():
    print("--- Starting Stage 3: Rewriting (HF Streaming) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(REJECT_LIST):
        print(f"Error: Reject list not found at {REJECT_LIST}")
        return False

    args = [(i, NUM_PROCESSES) for i in range(NUM_PROCESSES)]

    with Pool(
        NUM_PROCESSES,
        initializer=init_worker_rewrite,
        initargs=(REJECT_LIST,),
    ) as pool:
        pool.starmap(process_shard_rewrite, args)

    return True


def main():
    stages = [
        (run_stage_1_hashing, "Stage 1 (Hashing) failed."),
        (run_stage_2_deduplication, "Stage 2 (Dedup) failed."),
        (run_stage_3_rewriting, "Stage 3 (Rewriting) failed."),
    ]

    for stage_fn, err_msg in stages:
        if not stage_fn():
            print(err_msg)
            return


if __name__ == "__main__":
    main()
