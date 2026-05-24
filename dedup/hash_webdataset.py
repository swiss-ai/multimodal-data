import glob
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
import webdataset as wds

INPUT_WDS_GLOB = "/path/to/data/vision-datasets/LAION-Aesthetics/*.tar"
OUTPUT_DIR = "/tmp/toolbox/deduplicate_stage_1/results/laion"
NUM_CPUS = 280


def cv2_decoder(key, data):
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
    phash = cv2.img_hash.pHash(resized)  # type:ignore
    return phash.tobytes()


def process_shard_subset(shard_list, worker_id):
    ds = (
        wds.WebDataset(shard_list, shardshuffle=False)  # type:ignore
        .decode(cv2_decoder)
        .to_tuple("__key__", "jpg", "json")
    )

    print(f"Worker {worker_id} processing {len(shard_list)} shards...")

    record_count = 0
    records = []

    for key, img, meta in ds:
        _ = key
        h = compute_phash(img)
        records.append({"key": meta["url"], "orig_key": key, "phash": h})

        if len(records) >= 1000:
            fn = f"hashes_worker_{worker_id:03}_{record_count:02}.parquet"
            output_path = os.path.join(OUTPUT_DIR, fn)

            df = pd.DataFrame(records)
            df.to_parquet(output_path)

            record_count += 1
            records = []

    if records:
        fn = f"hashes_worker_{worker_id}_{record_count}_end.parquet"
        output_path = os.path.join(OUTPUT_DIR, fn)

        df = pd.DataFrame(records)
        df.to_parquet(output_path)

    return 1


def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Output directory {OUTPUT_DIR} already exists. Please remove it first.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_shards = sorted(glob.glob(INPUT_WDS_GLOB))

    chunks = np.array_split(all_shards, NUM_CPUS)
    chunks = [list(c) for c in chunks if len(c) > 0]

    print(f"Processing {len(all_shards)} shards across {len(chunks)} workers...")

    with Pool(len(chunks)) as pool:
        args = [(chunk, i) for i, chunk in enumerate(chunks)]
        results = pool.starmap(process_shard_subset, args)

    print(f"Done. Total chunks processed: {sum(results)}")


if __name__ == "__main__":
    main()
