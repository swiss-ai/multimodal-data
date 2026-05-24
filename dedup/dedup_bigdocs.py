import argparse
import glob
import os
from multiprocessing import Pool

# fmt:off
# Remove this line - set HF_TOKEN env var before running
os.environ.setdefault("HF_HOME", os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface")))
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
# fmt:on

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import webdataset as wds
from rocksdict import Options, Rdict, WriteBatch

parser = argparse.ArgumentParser()
parser.add_argument("stage", type=int)
parser.add_argument("num_processes", type=int)
args = parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = "/path/to/data/vision-datasets/hf_hub_cache/datasets--ServiceNow--BigDocs-7.5M/snapshots/dae4403c28307bd5328920740e81ce5232819e74"
ROOT_OUTPUT_DIR = "/path/to/data/vision-datasets/ServiceNow"
RESULT_DIR = "/tmp/bigdocs"
DB_PATH = os.path.join(RESULT_DIR, "db")
HASH_DIR = os.path.join(RESULT_DIR, "hash")
REJECT_LIST = os.path.join(RESULT_DIR, "reject.csv")
OUTPUT_DIR = os.path.join(ROOT_OUTPUT_DIR, "BigDocs-7.5M")

# Resources
INVALID_IMAGE = "INVALID"
NUM_PROCESSES = args.num_processes
ROCKSDB_BG_JOBS = 4

# Global variable for Stage 3 workers
reject_set = None

# ==========================================
# STAGE 1: HASHING
# ==========================================


def get_parquet_paths():
    paths = glob.glob(f"{INPUT_DIR}/**/*.parquet", recursive=True)
    return sorted(paths)


def compute_phash(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    resized = cv2.resize(gray, (32, 32))
    phash = cv2.img_hash.pHash(resized)
    return phash.tobytes().hex()


def process_shard_hashing(parquet_paths, wi):
    print(f"[Stage 1][{wi=}] starting stream...")

    records = []
    for path in parquet_paths:
        schema = pq.read_schema(path)
        if "image" not in schema.names:
            print(f"[Stage 1][{wi=}] skipping {os.path.basename(path)} (no image)")
            continue
        df = pd.read_parquet(path, columns=["sample_id", "image"])
        for _, row in df.iterrows():
            key = row["sample_id"]
            img_bytes = row["image"]["bytes"]

            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # validation
            if img is None:
                records.append({"key": key, "hash": INVALID_IMAGE})
                continue
            height, width = img.shape[:2]
            if not (100 <= width <= 8000 and 100 <= height <= 8000):
                records.append({"key": key, "hash": INVALID_IMAGE})
                continue

            hash = compute_phash(img)
            records.append({"key": key, "hash": hash})

    fn = f"hashes_{wi}_end.parquet"
    output_path = os.path.join(HASH_DIR, fn)
    pd.DataFrame(records).to_parquet(output_path)

    print(f"[Stage 1][{wi=}] finished processing")
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
    seen_keys = set()
    total_new = 0
    total_dupes = 0

    print(f"Scanning {len(hash_files)} hash files against persistent DB...")

    for hi, f in enumerate(hash_files):
        print(f"Processing file {hi + 1}/{len(hash_files)}: {os.path.basename(f)}")

        df = pd.read_parquet(f)
        wb = WriteBatch()
        seen_in_batch = set()
        batch_writes = 0

        if len(df) == 0:
            continue

        keys = df["key"].values
        hashes = df["hash"].values

        for key, img_hash in zip(keys, hashes):
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if img_hash == INVALID_IMAGE:
                duplicates.append(key)
                total_dupes += 1
                continue
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


def process_parquet_rewrite(args):
    parquet_path, idx = args
    global reject_set
    assert reject_set is not None

    worker_id = os.getpid()
    rel_path = os.path.relpath(parquet_path, INPUT_DIR)
    tar_name = f"{idx:05d}.tar"
    output_path = os.path.join(OUTPUT_DIR, tar_name)

    if os.path.exists(output_path):
        print(f"[Stage 3] Output exists, skipping: {rel_path}")
        return

    schema = pq.read_schema(parquet_path)
    if "image" not in schema.names:
        print(f"[Stage 3][{worker_id=}] skipping {rel_path} (no image column)")
        return

    print(f"[Stage 3][{worker_id=}] {rel_path}...")

    df = pd.read_parquet(parquet_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sink = wds.TarWriter(output_path)
    kept = 0
    dropped = 0

    for _, row in df.iterrows():
        key = row["sample_id"]
        if key in reject_set:
            dropped += 1
            continue

        img_bytes = row["image"]["bytes"]
        wds_key = key.replace(".", "_")
        sink.write({"__key__": wds_key, "jpg": img_bytes})
        kept += 1

    sink.close()

    # delete if output is empty
    if kept == 0:
        os.remove(output_path)

    return f"Worker {worker_id}: Kept {kept}, Dropped {dropped} in {rel_path}"


# ==========================================
# MAIN
# ==========================================


def run_stage_1_hashing():
    print("--- Starting Stage 1: Hashing ---")
    os.makedirs(HASH_DIR, exist_ok=True)

    parquet_paths = get_parquet_paths()
    print(f"Total parquet files: {len(parquet_paths)}")
    my_args = np.array_split(parquet_paths, NUM_PROCESSES)
    my_args = [(args, wi) for wi, args in enumerate(my_args)]

    with Pool(NUM_PROCESSES) as pool:
        pool.starmap(process_shard_hashing, my_args)

    return True


def run_stage_3_rewriting():
    print("--- Starting Stage 3: Rewriting ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(REJECT_LIST):
        print(f"Error: Reject list not found at {REJECT_LIST}")
        return False

    parquet_paths = get_parquet_paths()
    parquet_args = [(p, i) for i, p in enumerate(parquet_paths)]

    with Pool(
        NUM_PROCESSES,
        initializer=init_worker_rewrite,
        initargs=(REJECT_LIST,),
    ) as pool:
        for result in pool.imap_unordered(process_parquet_rewrite, parquet_args):
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


if __name__ == "__main__":
    print("Configuration:")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    main()
