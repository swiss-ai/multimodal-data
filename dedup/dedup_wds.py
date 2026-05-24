import glob
import multiprocessing
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import webdataset as wds
from rocksdict import Options, Rdict

# ==========================================
#               CONFIGURATION
# ==========================================

# Paths
INPUT_GLOB = "/path/to/data/vision-datasets/LAION-Aesthetics/*.tar"
INTERMEDIATE_DIR = "/tmp/toolbox/deduplicate_stage_1/results/laion"
DB_PATH = "/tmp/toolbox/deduplicate_stage_1/results/db/rocksdb"
REJECT_LIST_PATH = os.path.join(INTERMEDIATE_DIR, "reject_list.txt")
OUTPUT_DIR = "/path/to/data/vision-datasets/LAION-Aesthetics_clean"

# Resources
NUM_PROCESSES = 280
ROCKSDB_BG_JOBS = 8

reject_set = None


# ==========================================
#        STAGE 1 HELPER FUNCTIONS
#        (Hashing & Parquet Gen)
# ==========================================


def hashing_decoder(key, data):
    extension = key.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        return None
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def compute_phash(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    resized = cv2.resize(gray, (32, 32))
    phash = cv2.img_hash.pHash(resized)
    return phash.tobytes()


def process_shard_hashing(shard_list, worker_id):
    ds = (
        wds.WebDataset(shard_list, shardshuffle=False)  # type:ignore
        .decode(hashing_decoder)
        .to_tuple("__key__", "jpg", "json")
    )

    print(f"[Stage 1] Worker {worker_id} processing {len(shard_list)} shards...")

    record_count = 0
    records = []

    for key, img, meta in ds:
        h = compute_phash(img)
        records.append({"key": meta["url"], "orig_key": key, "phash": h})

        if len(records) >= 100000:
            fn = f"hashes_worker_{worker_id:03}_{record_count:02}.parquet"
            output_path = os.path.join(INTERMEDIATE_DIR, fn)

            df = pd.DataFrame(records)
            df.to_parquet(output_path)

            record_count += 1
            records = []

    if records:
        fn = f"hashes_worker_{worker_id}_{record_count}_end.parquet"
        output_path = os.path.join(INTERMEDIATE_DIR, fn)
        df = pd.DataFrame(records)
        df.to_parquet(output_path)


# ==========================================
#        STAGE 3 HELPER FUNCTIONS
#        (Filtering & Rewriting)
# ==========================================


def passthrough_decoder(key, data):
    extension = key.split(".")[-1].lower()
    if extension in ["jpg", "jpeg", "png"]:
        return data
    return None


def load_reject_list(path):
    s = set()
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                ko, url = parts[0], parts[1]
                s.add((ko, url))
    return s


def init_worker_rewrite(reject_path):
    global reject_set
    reject_set = load_reject_list(reject_path)


def process_shard_rewrite(shard_path):
    global reject_set
    basename = os.path.basename(shard_path)
    output_path = os.path.join(OUTPUT_DIR, basename)

    if os.path.exists(output_path):
        print(f"[Stage 3] Output exists, skipping: {basename}")
        return "Skipped"

    ds = (
        wds.WebDataset(shard_path, shardshuffle=False)  # type:ignore
        .decode(passthrough_decoder)
        .to_tuple("__key__", "jpg", "json")
    )

    sink = wds.TarWriter(output_path)  # type:ignore

    kept = 0
    dropped = 0

    for key, img_bytes, meta in ds:
        url = meta["url"]
        if (key, url) in reject_set:  # type:ignore
            dropped += 1
            continue

        sample = {"__key__": key, "json": meta, "jpg": img_bytes}
        sink.write(sample)
        kept += 1

    sink.close()
    return f"Kept: {kept}, Dropped: {dropped}"


# ==========================================
#             MAIN STAGES
# ==========================================


def run_stage_1_hashing():
    print("--- Starting Stage 1: Hashing & Parquet Generation ---")

    if os.path.exists(INTERMEDIATE_DIR):
        raise FileExistsError(f"Intermediate directory {INTERMEDIATE_DIR} already exists.")

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    all_shards = sorted(glob.glob(INPUT_GLOB))
    chunks = np.array_split(all_shards, NUM_PROCESSES)
    chunks = [list(c) for c in chunks if len(c) > 0]

    print(f"Processing {len(all_shards)} shards across {len(chunks)} workers...")

    with Pool(len(chunks)) as pool:
        args = [(chunk, i) for i, chunk in enumerate(chunks)]
        pool.starmap(process_shard_hashing, args)

    print("Stage 1 Done.")


def run_stage_2_deduplication():
    print("--- Starting Stage 2: Database Check & Reject List Gen ---")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    opts = Options()
    opts.create_if_missing(True)
    opts.set_max_background_jobs(ROCKSDB_BG_JOBS)

    db = Rdict(DB_PATH, options=opts)

    hash_files = sorted(glob.glob(f"{INTERMEDIATE_DIR}/*.parquet"))
    duplicates = []
    total_new = 0
    total_dupes = 0

    print(f"Scanning {len(hash_files)} hash files against persistent DB...")

    for fi, f in enumerate(hash_files):
        print(f"Processing file {fi + 1}/{len(hash_files)}")

        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            img_hash = row["phash"]
            key = row["key"]
            key_orig = row["orig_key"]

            if img_hash in db:
                duplicates.append((key, key_orig))
                total_dupes += 1
            else:
                db[img_hash] = key.encode("utf-8")  # type:ignore
                total_new += 1

    print("Finished scanning. Closing DB...")
    db.close()

    print(f"Saving {len(duplicates)} duplicate URLs to {REJECT_LIST_PATH}...")
    with open(REJECT_LIST_PATH, "w") as f:
        for key, ko in duplicates:
            f.write(f"{ko} {key}\n")

    print(f"Stage 2 Done. New items: {total_new}, Duplicates found: {total_dupes}")


def run_stage_3_rewriting():
    print("--- Starting Stage 3: Filtering & Rewriting Shards ---")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_shards = sorted(glob.glob(INPUT_GLOB))

    print(f"Rewriting {len(all_shards)} shards...")

    if not os.path.exists(REJECT_LIST_PATH):
        print(f"Error: Reject list not found at {REJECT_LIST_PATH}. Did Stage 2 run?")
        return

    # Using multiprocessing with initializer to load the set into memory once per worker
    with multiprocessing.Pool(NUM_PROCESSES, initializer=init_worker_rewrite, initargs=(REJECT_LIST_PATH,)) as pool:
        pool.map(process_shard_rewrite, all_shards)

    print("Stage 3 Done. Rewrite complete.")


def main():
    for si, stage_fn in enumerate([run_stage_1_hashing, run_stage_2_deduplication, run_stage_3_rewriting]):
        if not stage_fn():
            print(f"Stage {si + 1} failed. Aborting pipeline.")
            break


if __name__ == "__main__":
    main()
