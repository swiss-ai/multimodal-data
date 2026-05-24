import argparse
import glob
import os
from multiprocessing import Pool

import cv2
import pandas as pd
import webdataset as wds
from rocksdict import AccessType, Options, Rdict, WriteBatch

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int)
parser.add_argument("--num_processes", type=int)
parser.add_argument("--array-id", type=int, default=0)
parser.add_argument("--array-total", type=int, default=1)
args = parser.parse_args()

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_DIR = "/path/to/data/vision-datasets/MINT_1T"
SCRATCH_DIR = "/tmp/mint2"

ROOT_OUTPUT_DIR = "/path/to/data/vision-datasets/mlfoundations"
OUTPUT_DIR = os.path.join(ROOT_OUTPUT_DIR, "MINT-1T-HTML")

# STAGE 1
HASH_DIR = "/tmp/mint/hash/html"

# STAGE 2
DEDUP_DB_DIR = os.path.join(SCRATCH_DIR, "dedup_dbs")
REJECT_LISTS_DIR = os.path.join(SCRATCH_DIR, "reject_lists")

REJECT_DB = os.path.join(SCRATCH_DIR, "reject_db")

NUM_PROCESSES = args.num_processes
CHUNK_SIZE = 10
INVALID_IMAGE = "INVALID"
ROCKSDB_BG_JOBS = 4

reject_db = None


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


# def process_shard_hashing(paths, chunk_num):
#     if not paths:
#         return True
#
#     pid = os.getpid()
#     print(f"[Stage 1][{pid=},{chunk_num=}] starting stream...")
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
#     print(f"[Stage 1][{pid=},{chunk_num=}] done")
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
    worker_id, hash_files_slice = worker_args
    pid = os.getpid()

    db_path = os.path.join(DEDUP_DB_DIR, f"{worker_id:04d}")
    reject_path = os.path.join(REJECT_LISTS_DIR, f"{worker_id:04d}.txt")
    os.makedirs(DEDUP_DB_DIR, exist_ok=True)
    os.makedirs(REJECT_LISTS_DIR, exist_ok=True)

    db = make_rocksdb(db_path)
    total_new = 0
    total_dupes = 0

    with open(reject_path, "w") as reject_f:
        for hi, f in enumerate(hash_files_slice):
            print(f"[Stage 2][{pid=},w={worker_id}] {hi + 1}/{len(hash_files_slice)}: {os.path.basename(f)}")

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
                    wb[img_hash] = key.encode("utf-8")
                    seen_in_batch.add(img_hash)
                    batch_writes += 1
                    total_new += 1

            if batch_writes > 0:
                db.write(wb)

    db.close()
    print(f"[Stage 2][{pid=},w={worker_id}] Done. new={total_new}, dupes={total_dupes}")
    return (total_new, total_dupes)


def build_reject_db():
    """Merge per-worker reject files into a single reject RocksDB for stage 3."""
    print("--- Building reject DB ---")
    reject_files = sorted(glob.glob(os.path.join(REJECT_LISTS_DIR, "reject_*.txt")))
    print(f"Found {len(reject_files)} reject files")

    db = make_rocksdb(REJECT_DB, write_buffer_mb=256)
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
    print("--- Stage 2: Parallel Deduplication ---")

    hash_files = sorted(glob.glob(f"{HASH_DIR}/*.parquet"))
    print(f"Found {len(hash_files)} hash files, {NUM_PROCESSES} workers")

    worker_files = [[] for _ in range(NUM_PROCESSES)]
    for i, f in enumerate(hash_files):
        worker_files[i % NUM_PROCESSES].append(f)

    worker_args = [(i, wf) for i, wf in enumerate(worker_files) if wf]

    with Pool(NUM_PROCESSES) as pool:
        results = pool.map(process_hash_files_dedup, worker_args)

    total_new = sum(r[0] for r in results)
    total_dupes = sum(r[1] for r in results)
    print(f"Dedup done. new={total_new}, dupes={total_dupes}")

    # build_reject_db()
    return True


# ==========================================
# STAGE 3: REWRITING
# ==========================================


def init_worker_rewrite():
    global reject_db
    reject_db = Rdict(REJECT_DB, access_type=AccessType.read_only())


def process_tar_file_rewrite(tar_path):
    global reject_db
    pid = os.getpid()
    tar_filename = os.path.basename(tar_path)
    output_path = os.path.join(OUTPUT_DIR, tar_filename)

    if os.path.exists(output_path):
        return

    if not os.path.exists(tar_path):
        print(f"[Stage 3][{pid=}] not found: {tar_path}")
        return

    print(f"[Stage 3][{pid=}] {tar_filename}...")

    ds = wds.WebDataset([tar_path])
    sink = wds.TarWriter(output_path)
    kept = 0
    dropped = 0

    for sample in ds:
        composite_key = f"{sample['__url__']}:{sample['__key__']}"
        if composite_key in reject_db:
            dropped += 1
            continue
        sink.write({"__key__": sample["__key__"], "jpg": sample["jpg"]})
        kept += 1

    sink.close()

    if kept == 0:
        os.remove(output_path)

    return f"[Stage 3][{pid=}] Kept {kept}, Dropped {dropped} in {tar_filename}"


def run_stage_3():
    print("--- Stage 3: Rewriting ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(REJECT_DB):
        print(f"Error: Reject DB not found at {REJECT_DB}")
        return False

    tar_paths = get_tar_paths()
    print(f"Processing {len(tar_paths)} tar files, {NUM_PROCESSES} workers")

    with Pool(NUM_PROCESSES, initializer=init_worker_rewrite) as pool:
        for result in pool.imap_unordered(process_tar_file_rewrite, tar_paths):
            if result is not None:
                print(result)

    return True


# ==========================================
# MAIN
# ==========================================


def main():
    stages = {
        # 1: run_stage_1,
        2: run_stage_2,
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
