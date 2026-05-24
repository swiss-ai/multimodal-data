#!/usr/bin/env python3
"""Extract 5 WebDataset keys from each tar file and save to logs/{run_id}/."""

import argparse
import os
from datetime import datetime

import webdataset as wds

NUM_KEYS = 20


def extract_samples(tar_path, out_dir, n=NUM_KEYS):
    """Extract the first n WDS keys from tar_path into out_dir. Returns list of keys."""
    os.makedirs(out_dir, exist_ok=True)

    dataset = wds.WebDataset([tar_path], shardshuffle=False)
    for _, sample in zip(range(n), dataset):
        key = sample["__key__"]
        for field, value in sample.items():
            if field.startswith("__"):
                continue
            dest = os.path.join(out_dir, f"{key}.{field}")
            with open(dest, "wb") as f:
                f.write(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tar_files", nargs="+", help="Path(s) to tar file(s)")
    parser.add_argument("--num-keys", type=int, default=NUM_KEYS, help="Number of keys to extract per tar")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_root = os.path.join("logs", run_id)
    os.makedirs(logs_root, exist_ok=True)

    print(f"Run ID: {run_id}")
    print(f"Output: {logs_root}\n")

    for tar_path in args.tar_files:
        label = os.path.basename(os.path.dirname(tar_path))
        out_dir = os.path.join(logs_root, label)
        print(f"[{label}] {tar_path}")
        extract_samples(tar_path, out_dir, n=args.num_keys)


if __name__ == "__main__":
    main()
