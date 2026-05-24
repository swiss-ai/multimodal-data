import glob
import multiprocessing
import os

import webdataset as wds

INPUT_SHARDS = "/path/to/data/vision-datasets/LAION-Aesthetics/*.tar"
OUTPUT_DIR = "/path/to/data/vision-datasets/LAION-Aesthetics_clean"
REJECT_LIST_PATH = "/tmp/toolbox/deduplicate_stage_1/results/laion/reject_list.txt"
NUM_WORKERS = 250

reject_set = None


def custom_decoder(key, data):
    extension = key.split(".")[-1].lower()
    if extension in ["jpg", "jpeg", "png"]:
        return data
    return None


def load_reject_list(path):
    s = set()
    with open(path, "r") as f:
        for line in f:
            ko, url = line.strip().split()
            s.add((ko, url))
    return s


def init_worker(reject_path):
    global reject_set
    reject_set = load_reject_list(reject_path)


def process_shard(shard_path):
    global reject_set
    basename = os.path.basename(shard_path)
    output_path = os.path.join(OUTPUT_DIR, basename)
    if os.path.exists(output_path):
        print("Output already exists, skipping:", basename)
        return "Skipped"

    ds = (
        wds.WebDataset(shard_path, shardshuffle=False)  # type:ignore
        .decode(custom_decoder)
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
    print(f"Finished {basename}: Kept {kept}, Dropped {dropped}")
    return f"Kept: {kept}, Dropped: {dropped}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_shards = sorted(glob.glob(INPUT_SHARDS))

    print(f"Rewriting {len(all_shards)} shards...")

    with multiprocessing.Pool(NUM_WORKERS, initializer=init_worker, initargs=(REJECT_LIST_PATH,)) as pool:
        pool.map(process_shard, all_shards)

    print("Done rewriting.")


if __name__ == "__main__":
    main()
