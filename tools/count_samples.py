#!/usr/bin/env python3
"""Count the number of files inside each tar.gz archive using multiprocessing."""

import glob
import tarfile
from multiprocessing import Pool


def count_files_in_tar(tar_path):
    """Open a tar.gz and count how many files are inside it."""
    print(f"Processing {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            count = sum(1 for member in tf.getmembers() if member.isfile())
        return tar_path, count
    except Exception as e:
        return tar_path, f"ERROR: {e}"


if __name__ == "__main__":
    tar_files = sorted(glob.glob("*.tar.gz"))
    print(f"Found {len(tar_files)} tar.gz archives\n")

    with Pool(63) as pool:
        results = pool.map(count_files_in_tar, tar_files)

    total = 0
    for path, count in sorted(results):
        print(f"{path}: {count} files inside")
        if isinstance(count, int):
            total += count

    print(f"\nTotal files across all archives: {total}")
