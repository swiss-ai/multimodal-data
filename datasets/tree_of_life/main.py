import argparse
import glob
import os
import pickle
from multiprocessing import Pool

# Set these environment variables before running:
#   export HF_TOKEN="your_token"
#   export HF_HOME="/path/to/hf_cache"
os.environ.setdefault(
    "HF_HOME",
    os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface")),
)
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"])

import cv2
import numpy as np
import pandas as pd
import webdataset as wds
from rocksdict import Options, Rdict, WriteBatch

parser = argparse.ArgumentParser()
parser.add_argument("stage", type=int)
parser.add_argument("num_processes", type=int)
args = parser.parse_args()

print("Configuration:")
for arg_name, arg_value in vars(args).items():
    print(f"{arg_name}: {arg_value}")

# Configuration

INPUT_DIR = "/path/to/data/vision-datasets/hf_hub_cache/datasets--imageomics--TreeOfLife-10M/snapshots/91debffb7146c32c89d76feb1eb575b555e2ecc7/dataset"
ROOT_OUTPUT_DIR = "/path/to/data/vision-datasets/imageomics"
RESULT_DIR = "/tmp/tol"
DB_PATH = os.path.join(RESULT_DIR, "db")
HASH_DIR = os.path.join(RESULT_DIR, "hash")
REJECT_LIST = os.path.join(RESULT_DIR, "reject.csv")
OUTPUT_DIR = os.path.join(ROOT_OUTPUT_DIR, "TreeOfLife-10M")

# Resources
INVALID_IMAGE = "INVALID"
NUM_PROCESSES = args.num_processes
ROCKSDB_BG_JOBS = 4

# Global variable for Stage 3 workers
LICENSE_DICT_PATH = "data/license_dict.pkl"
reject_set = None
license_mapping = None

allowed_license = [
    "cc-0-1.0",
    "cc-by",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-publicdomain",
]
not_allowed_license = [
    "cc-by-nc",
    "cc-by-nc-2.0",
    "cc-by-nc-4.0",
    "cc-by-nc-sa",
    "cc-by-nc-sa-2.5",
    "cc-by-nc-sa-3.0",
    "cc-by-nc-sa-4.0",
    "cc-by-sa-2.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "No known copyright restrictions",
]

# Stage 1: Hashing


def get_tar_paths():
    paths = glob.glob(f"{INPUT_DIR}/**/*.tar.gz")
    return sorted(paths)


def compute_phash(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    resized = cv2.resize(gray, (32, 32))
    phash = cv2.img_hash.pHash(resized)
    return phash.tobytes().hex()


def process_shard_hashing(paths, wi):
    global license_mapping
    assert license_mapping is not None

    print(f"[Stage 1][{wi=}] starting stream...")

    ds = wds.WebDataset(paths, shardshuffle=False).to_tuple("__key__", "jpg")
    # ds = ds.slice(10)

    records = []
    for key, img_bytes in ds:
        license_name = license_mapping[key]
        if license_name not in allowed_license:
            assert license_name in not_allowed_license
            records.append({"key": key, "hash": INVALID_IMAGE})
            continue

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # validation
        if img is None:
            records.append({"key": key, "hash": INVALID_IMAGE})
            continue
        height, width = img.shape[:2]
        if not (200 <= width <= 8000 and 200 <= height <= 8000):
            records.append({"key": key, "hash": INVALID_IMAGE})
            continue

        hash = compute_phash(img)
        records.append({"key": key, "hash": hash})

    fn = f"hashes_{wi}_end.parquet"
    output_path = os.path.join(HASH_DIR, fn)
    pd.DataFrame(records).to_parquet(output_path)

    print(f"[Stage 1][{wi=}] finished processing")
    return True


# Stage 2: Deduplication


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


# Stage 3: Rewriting


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
    output_path = os.path.join(OUTPUT_DIR, tar_filename)
    if os.path.exists(output_path):
        print(f"[Stage 3] Output exists, skipping: {tar_filename}")
        return

    print(f"[Stage 3][{worker_id=}] {tar_filename}...")

    if not os.path.exists(tar_path):
        print(f"[Stage 3][{worker_id=}] shard not found {tar_path}")
        return

    ds = wds.WebDataset([tar_path], shardshuffle=False).to_tuple("__key__", "jpg")
    # ds = ds.slice(10)

    sink = wds.TarWriter(output_path)
    kept = 0
    dropped = 0

    for key, img_bytes in ds:
        if key in reject_set:
            dropped += 1
            continue

        sink.write({"__key__": key, "jpg": img_bytes})
        kept += 1

    sink.close()
    return f"Worker {worker_id}: Kept {kept}, Dropped {dropped} in {tar_filename}"


# Main


def init_worker_hashing(license_dict_path):
    global license_mapping
    with open(license_dict_path, "rb") as f:
        license_mapping = pickle.load(f)


def run_stage_1_hashing():
    print("--- Starting Stage 1: Hashing ---")
    os.makedirs(HASH_DIR, exist_ok=True)

    tar_paths = get_tar_paths()
    my_args = np.array_split(tar_paths, NUM_PROCESSES)

    print(f"Total tar files: {len(tar_paths)}")
    for i, args in enumerate(my_args):
        print(f"Worker {i}: {len(args)} tar files.", end=" ")
        print(f": {os.path.basename(args[0])} ... {os.path.basename(args[-1])}")
    my_args = [(args, wi) for wi, args in enumerate(my_args)]

    with Pool(
        NUM_PROCESSES,
        initializer=init_worker_hashing,
        initargs=(LICENSE_DICT_PATH,),
    ) as pool:
        pool.starmap(process_shard_hashing, my_args)

    return True


def run_stage_3_rewriting():
    print("--- Starting Stage 3: Rewriting ---")
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


if __name__ == "__main__":
    main()
