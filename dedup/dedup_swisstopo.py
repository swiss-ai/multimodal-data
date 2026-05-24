import argparse
import glob
import os
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

parser = argparse.ArgumentParser()
parser.add_argument("stage", type=int)
parser.add_argument("num_processes", type=int)
args = parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = "/path/to/data/vision-datasets/SWISSIMAGE"
OUTPUT_DIR = "/path/to/data/vision-datasets/SWISSIMAGE3"
RESULT_DIR = "/tmp/swissimage3"
DB_PATH = os.path.join(RESULT_DIR, "db")
HASH_DIR = os.path.join(RESULT_DIR, "hash")
REJECT_LIST = os.path.join(RESULT_DIR, "reject.parquet")

# Resources
INVALID_IMAGE = "INVALID"
NUM_PROCESSES = args.num_processes
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
    resized = cv2.resize(gray, (8, 8))
    ahash = cv2.img_hash.pHash(resized)
    return ahash.tobytes().hex()


def process_shard_hashing(paths, wi):
    print(f"[Stage 1][{wi=}] starting stream...")

    ds = wds.WebDataset(paths, shardshuffle=False).to_tuple("__url__", "__key__", "jpg")

    records = []
    for _url_full, _key, img_bytes in ds:
        _url = os.path.basename(_url_full)

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        assert img.shape == (10000, 10000, 3)
        for x in range(10):
            for y in range(10):
                x_start, x_end = x * 1000, (x + 1) * 1000
                y_start, y_end = y * 1000, (y + 1) * 1000
                tile = img[y_start:y_end, x_start:x_end]

                hash = compute_phash(tile)
                key = f"{_url}--{_key}--tile_{x}_{y}"
                records.append({"key": key, "hash": hash})

    fn = f"{wi:05}.parquet"
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
    pd.DataFrame({"key": duplicates}).to_parquet(REJECT_LIST)

    print(f"Stage 2 Done. New items: {total_new}, Duplicates found: {total_dupes}")
    return True


# ==========================================
# STAGE 3: REWRITING
# ==========================================


def load_reject_list(path):
    df = pd.read_parquet(path)
    return set(df["key"].values)


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

    sink = wds.TarWriter(output_path)
    kept = 0
    dropped = 0

    for key, img_bytes in ds:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        assert img.shape == (10000, 10000, 3)
        for x in range(10):
            for y in range(10):
                tile_key = f"{tar_filename}--{key}--tile_{x}_{y}"
                if tile_key in reject_set:
                    dropped += 1
                    continue

                x_start, x_end = x * 1000, (x + 1) * 1000
                y_start, y_end = y * 1000, (y + 1) * 1000
                tile = img[y_start:y_end, x_start:x_end]

                tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                _, tile_bytes = cv2.imencode(".jpg", tile_bgr)

                sink_key = f"{key}_tile_{x}_{y}"
                sink.write({"__key__": sink_key, "jpg": tile_bytes.tobytes()})
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
    print(f"Total tar files: {len(tar_paths)}")

    args = []
    for i, arg_list in enumerate(np.array_split(tar_paths, NUM_PROCESSES)):
        print(f"Worker {i}: {len(arg_list)} tar files.", end=" ")
        print(f": {os.path.basename(arg_list[0])} ... {os.path.basename(arg_list[-1])}")
        args.append((arg_list.tolist(), i))

    with Pool(NUM_PROCESSES) as pool:
        pool.starmap(process_shard_hashing, args)

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
    print("Configuration:")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    main()
