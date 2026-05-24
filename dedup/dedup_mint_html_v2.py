import argparse
import gc
import glob
import json
import os
from multiprocessing import get_context

import cv2
import numpy as np
import pandas as pd
import webdataset as wds
from rocksdict import Options, Rdict, WriteBatch

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int)
parser.add_argument("--num_processes", type=int)
parser.add_argument("--array-id", type=int)
parser.add_argument("--array-total", type=int)
args = parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================

ROBOTS_DIR = "/path/to/scratch/snajemmeyer/MINT_1T_robots_filtered"
INPUT_DIR = "/path/to/data/vision-datasets/MINT_1T"
SCRATCH_DIR = "/tmp/mint"

# ROOT_OUTPUT_DIR = "/path/to/data/vision-datasets/mlfoundations"
ROOT_OUTPUT_DIR = "/tmp/shared"
OUTPUT_DIR = os.path.join(ROOT_OUTPUT_DIR, "MINT-1T-HTML")

# STAGE 1
HASH_DIR = os.path.join(SCRATCH_DIR, "hash")

# STAGE 2
DEDUP_DB_DIR = os.path.join(SCRATCH_DIR, "g_dedup_dbs")
REJECT_LISTS_DIR = os.path.join(SCRATCH_DIR, "g_reject_lists")

# STAGE 2/3
REJECT_DB = os.path.join(SCRATCH_DIR, "g_reject_db")

NUM_PROCESSES = args.num_processes
INVALID_IMAGE = "INVALID_IMAGE"
ROCKSDB_BG_JOBS = 4


# ==========================================
# HELPERS
# ==========================================


def get_tar_paths():
    return sorted(glob.glob(f"{INPUT_DIR}/**/*.tar"))


def compute_phash(image_array):
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array
    resized = cv2.resize(gray, (32, 32))
    return cv2.img_hash.pHash(resized).tobytes().hex()


def make_rocksdb(path, write_buffer_mb=64):
    opts = Options()
    opts.create_if_missing(True)
    opts.set_max_background_jobs(ROCKSDB_BG_JOBS)
    opts.set_write_buffer_size(write_buffer_mb * 1024 * 1024)
    return Rdict(path, options=opts)


# ==========================================
# STAGE 1: HASHING
# ==========================================


# NOTE: This stage is currently disabled. It was run once to generate the hash
# files, and the rest of the pipeline relies on those existing files.

# def process_shard_hashing(paths, chunk_num):
#     if not paths:
#         return True
#
#     pid = os.getpid()
#     print(f"[S1][{pid=},{chunk_num=}] starting stream...")
#
#     ds = wds.WebDataset(paths).decode()
#     records = []
#     part_idx = 0
#
#     for sample in ds:
#         key = f"{sample['__url__']}:{sample['__key__']}"
#         img = sample.get("jpg")
#
#         if img is None:
#             records.append({"key": key, "hash": INVALID_IMAGE})
#             continue
#
#         img = np.asarray(img)
#         if img.ndim < 2:
#             records.append({"key": key, "hash": INVALID_IMAGE})
#             continue
#         h, w = img.shape[:2]
#         if not (200 <= w <= 8000 and 200 <= h <= 8000):
#             records.append({"key": key, "hash": INVALID_IMAGE})
#             continue
#
#         records.append({"key": key, "hash": compute_phash(img)})
#
#         if len(records) >= 200000:
#             fn = f"hashes_{chunk_num:04d}_{part_idx:04d}.parquet"
#             pd.DataFrame(records).to_parquet(os.path.join(HASH_DIR, fn))
#             part_idx += 1
#             records = []
#
#     if records:
#         fn = f"hashes_{chunk_num:04d}_{part_idx:04d}_end.parquet"
#         pd.DataFrame(records).to_parquet(os.path.join(HASH_DIR, fn))
#
#     print(f"[S1][{pid=},{chunk_num=}] done")
#     return True
#
#
# def run_stage_1():
#     print("--- Stage 1: Hashing ---")
#     os.makedirs(HASH_DIR, exist_ok=True)
#
#     tar_paths = get_tar_paths()
#     chunks = [
#         tar_paths[i : i + CHUNK_SIZE] for i in range(0, len(tar_paths), CHUNK_SIZE)
#     ]
#     all_args = [(chunks[i], i) for i in range(len(chunks))]
#
#     # Slice for this SLURM array task
#     per_task = (len(all_args) + args.array_total - 1) // args.array_total
#     start = args.array_id * per_task
#     end = min(start + per_task, len(all_args))
#     my_args = all_args[start:end]
#     print(
#         f"Array {args.array_id}/{args.array_total}: chunks {start}-{end} of {len(all_args)}"
#     )
#
#     with Pool(NUM_PROCESSES) as pool:
#         pool.starmap(process_shard_hashing, my_args)
#
#     return True


# ==========================================
# STAGE 2: DEDUPLICATION (parallel, sharded)
# ==========================================


def process_hash_files_dedup(worker_args):
    wid, hash_files_slice = worker_args
    pid = os.getpid()

    db_path = os.path.join(DEDUP_DB_DIR, "global")
    reject_path = os.path.join(REJECT_LISTS_DIR, f"{wid:04d}.txt")
    os.makedirs(DEDUP_DB_DIR, exist_ok=True)
    os.makedirs(REJECT_LISTS_DIR, exist_ok=True)

    db = make_rocksdb(db_path)
    total_new = 0
    total_dupes = 0

    with open(reject_path, "w") as reject_f:
        for hi, f in enumerate(hash_files_slice):
            fbase = os.path.basename(f)
            print(f"[S2][{pid=},w={wid}] {hi + 1}/{len(hash_files_slice)}: {fbase}")

            df = pd.read_parquet(f)
            if len(df) == 0:
                continue

            wb = WriteBatch()
            seen_in_batch = set()
            batch_writes = 0

            for key, img_hash in zip(df["key"].values, df["hash"].values):
                if img_hash == INVALID_IMAGE:
                    reject_f.write(f"{key}\n")
                    total_dupes += 1
                    continue
                if img_hash in db or img_hash in seen_in_batch:
                    reject_f.write(f"{key}\n")
                    total_dupes += 1
                else:
                    wb[img_hash] = b""
                    seen_in_batch.add(img_hash)
                    batch_writes += 1
                    total_new += 1

            if batch_writes > 0:
                db.write(wb)

    db.close()
    print(f"[S2][{pid=},w={wid}] Done. new={total_new}, dupes={total_dupes}")
    return (total_new, total_dupes)


def build_reject_db():
    print("--- Building reject DB ---")
    reject_files = sorted(glob.glob(os.path.join(REJECT_LISTS_DIR, "*.txt")))
    print(f"Found {len(reject_files)} reject files")

    db = make_rocksdb(REJECT_DB, write_buffer_mb=512)
    total = 0

    for rf in reject_files:
        wb = WriteBatch()
        batch_size = 0
        with open(rf, "r") as f:
            for line in f:
                key = line.strip()
                if key:
                    wb[key] = b""
                    batch_size += 1
                    total += 1
                    if batch_size >= 500000:
                        db.write(wb)
                        wb = WriteBatch()
                        batch_size = 0
        if batch_size > 0:
            db.write(wb)
        print(f"  Loaded {os.path.basename(rf)}")

    db.close()
    print(f"Reject DB: {total} entries at {REJECT_DB}")


def run_stage_2():
    print("--- Stage 2: Global Deduplication ---")

    hash_files = sorted(glob.glob(f"{HASH_DIR}/*.parquet"))
    print(f"Found {len(hash_files)} hash files total")

    chunks = np.array_split(hash_files, args.array_total)
    my_hash_files = list(chunks[args.array_id])
    num_files, num_total = len(my_hash_files), len(hash_files)
    print(f"Array {args.array_id}/{args.array_total}: {num_files} files of {num_total}")

    result = process_hash_files_dedup((args.array_id, my_hash_files))
    print(f"Dedup done. new={result[0]}, dupes={result[1]}")

    if args.array_id == args.array_total - 1:
        build_reject_db()
    return True


# ==========================================
# STAGE 3: REWRITING
# ==========================================


reject_set = None


def init_worker_rewrite():
    gc.disable()


def process_tar_file_rewrite(task_args):
    tid, tar_path = int(task_args[0]), task_args[1]
    global reject_set
    pid = os.getpid()
    tar_filename = f"part-{tid:06d}.tar"
    output_path = os.path.join(OUTPUT_DIR, tar_filename)
    tmp_path = output_path + ".tmp"

    parquet_path = tar_path.replace(INPUT_DIR, ROBOTS_DIR).replace(".tar", ".parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        valid_urls = set(df["url"].values)
    else:
        valid_urls = None

    assert os.path.exists(tar_path)
    if os.path.exists(output_path):
        return f"[S3][{pid=}] SKIP (done) {tar_filename}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"[S3][{pid=}] {tar_filename} ({tar_path})...")

    try:
        ds = wds.WebDataset(
            [tar_path],
            shardshuffle=False,
            handler=wds.warn_and_continue,
        )
        sink = wds.TarWriter(tmp_path)
        kept = 0
        dropped = 0

        for sample in ds:
            json_data = json.loads(sample["json"].decode("utf-8"))
            url = json_data["url"]

            composite_key = f"{sample['__url__']}:{sample['__key__']}"
            if composite_key in reject_set or (valid_urls is not None and url not in valid_urls):
                dropped += 1
                continue
            sink.write(
                {
                    "__key__": sample["__key__"],
                    "jpg": sample["jpg"],
                    "json": {"url": url},
                }
            )
            kept += 1

        sink.close()

        if kept == 0:
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, output_path)

        return f"[S3][{pid=}] Kept {kept}, Dropped {dropped} in {tar_filename} ({tar_path})"
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"{tar_filename} ({tar_path}): {type(e).__name__}: {e}")


def run_stage_3():
    global reject_set
    print("--- Stage 3: Rewriting ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reject_files = sorted(glob.glob(os.path.join(REJECT_LISTS_DIR, "*.txt")))
    if not reject_files:
        print(f"Error: No reject files found in {REJECT_LISTS_DIR}")
        return False

    print(f"Loading reject keys from {len(reject_files)} files...")
    reject_set = set()
    for rf in reject_files:
        print(f"  Loading {os.path.basename(rf)}...")
        with open(rf, "r") as f:
            for line in f:
                key = line.strip()
                if key:
                    reject_set.add(key)
    print(f"Loaded {len(reject_set)} reject keys into memory")

    tar_paths = get_tar_paths()
    work_items = [(i, tar_path) for i, tar_path in enumerate(tar_paths)]
    print(f"Processing {len(tar_paths)} tar files, {NUM_PROCESSES} workers")

    work_items_chunked = np.array_split(work_items, args.array_total)
    work_items = list(work_items_chunked[args.array_id])

    ctx = get_context("fork")
    with ctx.Pool(NUM_PROCESSES, initializer=init_worker_rewrite) as pool:
        for result in pool.imap_unordered(process_tar_file_rewrite, work_items):
            if result is not None:
                print(result)

    return True


# ==========================================
# MAIN
# ==========================================


def main():
    stages = {
        # 1: run_stage_1,
        # 2: run_stage_2,
        3: run_stage_3,
    }
    fn = stages[args.stage]
    success = fn()
    print(f"Stage {args.stage} {'completed' if success else 'FAILED'}.")


if __name__ == "__main__":
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    main()
