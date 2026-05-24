#!/usr/bin/env python3
"""Extract all images for specific hashes from tar archives.

Usage: python3 check_duplicates.py <hash1> [hash2] [hash3] ...

Queries parquet files to find all keys matching the given hashes,
then extracts the images into per-hash directories under extracted/.
"""

import os
import sys
import tarfile
from collections import defaultdict

import duckdb

PARQUET_DIR = os.environ.get("PARQUET_DIR", "/tmp/mint/hash")


def query_by_hashes(parquet_dir, hashes):
    """Query all keys for specific hashes.

    Returns list of (hash, count, [key, ...]) tuples.
    """
    con = duckdb.connect(":memory:")
    hash_list = ", ".join(f"'{h}'" for h in hashes)
    rows = con.execute(f"""
        SELECT hash, COUNT(*) AS cnt, list(key) AS keys
        FROM '{parquet_dir}/*.parquet'
        WHERE hash IN ({hash_list})
        GROUP BY hash
    """).fetchall()
    con.close()
    return rows


def build_extraction_plan(rows):
    tar_files = defaultdict(list)
    for hash_val, _cnt, keys in rows:
        for key in keys:
            tar_path, base = key.rsplit(":", 1)
            member = f"{base}.jpg"
            tar_files[tar_path].append((member, hash_val, key))
    return tar_files


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hash1> [hash2] [hash3] ...")
        sys.exit(1)

    hashes = sys.argv[1:]
    outdir = "extracted"

    print(f"Querying all keys for {len(hashes)} hash(es) from {PARQUET_DIR} ...")
    rows = query_by_hashes(PARQUET_DIR, hashes)
    for hash_val, cnt, _keys in rows:
        print(f"  {hash_val}: {cnt} entries")
    not_found = set(hashes) - {r[0] for r in rows}
    for h in not_found:
        print(f"  {h}: not found")

    tar_files = build_extraction_plan(rows)

    for tar_path, members in tar_files.items():
        names = [m for m, _, _ in members]
        print(f"Extracting {names} from {tar_path} ...")
        with tarfile.open(tar_path) as tf:
            for member, hash_val, key in members:
                hash_dir = os.path.join(outdir, hash_val)
                os.makedirs(hash_dir, exist_ok=True)
                unique_name = key.replace("/", "_").replace(":", "-") + ".jpg"
                info = tf.getmember(member)
                with tf.extractfile(info) as src:
                    with open(os.path.join(hash_dir, unique_name), "wb") as dst:
                        dst.write(src.read())

    print(f"Done. Files extracted to {outdir}/")


if __name__ == "__main__":
    main()
