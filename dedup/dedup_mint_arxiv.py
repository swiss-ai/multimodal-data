import argparse
import glob
import os
import sys
from multiprocessing import Pool

# fmt:off
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

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, required=True)
args = parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = "/path/to/data/vision-datasets/hf_hub_cache/datasets--mlfoundations--MINT-1T-ArXiv/snapshots/7c5b00ffd5b563071010c3bf2082b4a8f836eb72"
RESULT_DIR = "/tmp/toolbox/deduplicate_stage_1/results"
DB_PATH = os.path.join(RESULT_DIR, "db_arxiv")
HASH_DIR = os.path.join(RESULT_DIR, "mint_1t_arxiv")
REJECT_LIST = os.path.join(HASH_DIR, "reject_list.txt")

ROOT_OUTPUT_DIR = "/path/to/data/vision-datasets/mlfoundations"
OUTPUT_DIR = os.path.join(ROOT_OUTPUT_DIR, "MINT-1T-ArXiv")

# Resources
NUM_PROCESSES = 128
ROCKSDB_BG_JOBS = 4

reject_set = None

# ==========================================
# STAGE 1: HASHING
# ==========================================


def get_tar_paths():
    paths = glob.glob(f"{INPUT_DIR}/*.tar")
    return sorted(paths)


def compute_phash(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    resized = cv2.resize(gray, (32, 32))
    phash = cv2.img_hash.pHash(resized)
    return phash.tobytes().hex()


def process_shard_hashing(paths, chunk_num):
    worker_id = os.getpid()
    print(f"[Stage 1][{worker_id=},{chunk_num=}] starting stream...")

    if not paths:
        print(f"[Stage 1][{worker_id=},{chunk_num=}] no shards to process")
        return True

    ds = load_dataset(
        "webdataset",
        data_files=paths,
        split="train",
        streaming=True,
    )
    ds = ds.cast_column("jpg", HFImage(decode=False))

    records = []
    part_idx = 0

    for item in ds:
        img_data = item["jpg"]
        img_bytes = img_data["bytes"]
        key = f"{item['__url__']}:{item['__key__']}"

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hash = compute_phash(img)
        records.append({"key": key, "hash": hash})

        if len(records) >= 200000:
            fn = f"hashes_{chunk_num:04d}_{part_idx:04d}.parquet"
            output_path = os.path.join(HASH_DIR, fn)
            pd.DataFrame(records).to_parquet(output_path)
            part_idx += 1
            records = []

    if records:
        fn = f"hashes_{chunk_num:04d}_{part_idx:04d}_end.parquet"
        output_path = os.path.join(HASH_DIR, fn)
        pd.DataFrame(records).to_parquet(output_path)

    print(f"[Stage 1][{worker_id=},{chunk_num=}] finished processing")
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


def process_tar_file_rewrite(tar_path):
    global reject_set
    assert reject_set is not None

    worker_id = os.getpid()
    tar_filename = os.path.basename(tar_path)
    output_filename = tar_filename
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    if os.path.exists(output_path):
        print(f"[Stage 3] Output exists, skipping: {output_filename}")
        return

    print(f"[Stage 3][{worker_id=}] {tar_filename} to {output_filename}...")

    if not os.path.exists(tar_path):
        print(f"[Stage 3][{worker_id=}] shard not found {tar_path}")
        return

    ds = load_dataset(
        "webdataset",
        data_files=[tar_path],
        split="train",
        streaming=True,
    )
    ds = ds.cast_column("jpg", HFImage(decode=False))

    sink = wds.TarWriter(output_path)
    kept = 0
    dropped = 0

    for item in ds:
        img_data = item["jpg"]
        img_bytes = img_data["bytes"]
        composite_key = f"{item['__url__']}:{item['__key__']}"
        original_key = item["__key__"]

        if composite_key in reject_set:
            dropped += 1
            continue

        sink.write({"__key__": original_key, "jpg": img_bytes})
        kept += 1

    sink.close()
    return f"Worker {worker_id}: Kept {kept}, Dropped {dropped} in {tar_filename}"


# ==========================================
# MAIN
# ==========================================


def run_stage_1_hashing():
    print("--- Starting Stage 1: Hashing ---")
    os.makedirs(HASH_DIR, exist_ok=True)

    tar_paths = get_tar_paths()
    if not tar_paths:
        print("No .tar files found in INPUT_DIR. Skipping hashing.")
        return True
    chunks = np.array_split(tar_paths, NUM_PROCESSES)
    my_args = [(chunk.tolist(), i) for i, chunk in enumerate(chunks)]

    with Pool(NUM_PROCESSES) as pool:
        pool.starmap(process_shard_hashing, my_args)

    return True


def run_stage_3_rewriting():
    print("--- Starting Stage 3: Rewriting (HF Streaming) ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(REJECT_LIST):
        print(f"Error: Reject list not found at {REJECT_LIST}")
        return False

    tar_paths = get_tar_paths()

    with Pool(
        NUM_PROCESSES,
        initializer=init_worker_rewrite,
        initargs=(REJECT_LIST,),
    ) as pool:
        for result in pool.imap_unordered(process_tar_file_rewrite, tar_paths):
            if result is not None:
                print(result)

    return True


def main():
    stages = {
        1: run_stage_1_hashing,
        2: run_stage_2_deduplication,
        3: run_stage_3_rewriting,
    }

    fn = stages[args.stage]
    success = fn()
    if success:
        print(f"Stage {args.stage} completed successfully.")
    else:
        print(f"Stage {args.stage} failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
