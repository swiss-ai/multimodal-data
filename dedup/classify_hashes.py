import glob
import os

import pandas as pd
from rocksdict import Options, Rdict

HASH_PARQUET_DIR = "/tmp/toolbox/deduplicate_stage_1/results/laion"
REJECT_LIST_OUTPUT = os.path.join(HASH_PARQUET_DIR, "reject_list.txt")
DB_PATH = "/tmp/toolbox/deduplicate_stage_1/results/db/rocksdb"


def main():
    opts = Options()
    opts.create_if_missing(True)
    opts.set_max_background_jobs(4)
    db = Rdict(DB_PATH, options=opts)

    hash_files = sorted(glob.glob(f"{HASH_PARQUET_DIR}/*.parquet"))
    duplicates = []
    total_new = 0
    total_dupes = 0

    print(f"Scanning {len(hash_files)} hash files against persistent DB...")

    for fi, f in enumerate(hash_files):
        print(f"Processing {fi + 1}/{len(hash_files)}", end="\r")
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

    print(f"Saving {len(duplicates)} duplicate URLs to {REJECT_LIST_OUTPUT}...")
    with open(REJECT_LIST_OUTPUT, "w") as f:
        for key, ko in duplicates:
            f.write(f"{ko} {key}\n")

    print(f"Done. New: {total_new}, Dupes: {total_dupes}")


if __name__ == "__main__":
    main()
