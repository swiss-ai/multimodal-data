#!/usr/bin/env python3
"""Extract duplicate files from tar archives, grouped by hash.

Queries parquet files to find the top N most duplicated hashes,
then extracts sample files from tar archives into per-hash directories.
"""

import os
import tarfile
from collections import defaultdict

import duckdb

PARQUET_DIR = os.environ.get("PARQUET_DIR", "/tmp/mint/hash")
TOP_N = 25
SAMPLES_PER_HASH = 10


def query_duplicates(parquet_dir, top_n, samples_per_hash):
    """Query parquet files for the top duplicated hashes and their sample keys.

    Returns list of (hash, duplicate_count, [key, ...]) tuples.
    """
    con = duckdb.connect(":memory:")
    rows = con.execute(f"""
        WITH top_hashes AS (
            SELECT hash, COUNT(*) AS duplicate_count
            FROM '{parquet_dir}/*.parquet'
            GROUP BY ALL
            ORDER BY duplicate_count DESC
            LIMIT {top_n}
        ),
        ranked_keys AS (
            SELECT
                p.hash,
                t.duplicate_count,
                p.key,
                ROW_NUMBER() OVER (PARTITION BY p.hash) AS rn
            FROM '{parquet_dir}/*.parquet' p
            JOIN top_hashes t ON p.hash = t.hash
        )
        SELECT hash, duplicate_count, list(key) AS sample_keys
        FROM ranked_keys
        WHERE rn <= {samples_per_hash}
        GROUP BY ALL
        ORDER BY duplicate_count DESC
    """).fetchall()
    con.close()
    return rows


def build_extraction_plan(rows):
    tar_files = defaultdict(list)
    for hash_val, _dup_count, sample_keys in rows:
        for key in sample_keys:
            tar_path, base = key.rsplit(":", 1)
            member = f"{base}.jpg"
            tar_files[tar_path].append((member, hash_val, key))
    return tar_files


def main():
    outdir = "extracted"

    print(f"Querying top {TOP_N} duplicated hashes from {PARQUET_DIR} ...")
    rows = query_duplicates(PARQUET_DIR, TOP_N, SAMPLES_PER_HASH)
    for hash_val, dup_count, sample_keys in rows:
        print(f"  {hash_val}: {dup_count} duplicates, {len(sample_keys)} samples")

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
                with (
                    tf.extractfile(info) as src,
                    open(os.path.join(hash_dir, unique_name), "wb") as dst,
                ):
                    dst.write(src.read())

    print(f"Done. Files extracted to {outdir}/")


if __name__ == "__main__":
    main()
