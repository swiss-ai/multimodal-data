#!/usr/bin/env python3
"""
Filter webdataset shards to keep only commercially-licensed samples.

Reads .tar shards from the HF hub cache, filters by allowed archive IDs,
and writes new .tar shards containing only licensed samples.

Uses multiprocessing for embarrassingly parallel execution.
"""

import argparse
import os
import shutil
import tarfile
from multiprocessing import Pool
from functools import partial

from tqdm import tqdm


def get_all_shard_paths(snapshot_dir):
    """Collect all .tar shard paths from audio/ and audio2/ under the snapshot."""
    shards = []
    for subdir in ("audio", "audio2"):
        d = os.path.join(snapshot_dir, subdir)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".tar"):
                shards.append((subdir, fname, os.path.join(d, fname)))
    return shards


def filter_shard(shard_info, allowed_ids, output_dir):
    """
    Filter a single .tar shard, keeping only allowed samples.

    Returns:
        (filename, total_members, kept_members, error)
    """
    subdir, fname, input_path = shard_info
    out_subdir = os.path.join(output_dir, subdir)
    out_path = os.path.join(out_subdir, fname)
    done_path = os.path.join(output_dir, ".done", subdir, fname + ".done")

    # Skip if already completed (resumable)
    if os.path.exists(done_path):
        return (fname, 0, 0, None)

    total = 0
    kept = 0

    try:
        with tarfile.open(input_path, "r") as tin:
            members_to_keep = []
            for member in tin.getmembers():
                total += 1
                key = member.name
                dot = key.rfind(".")
                if dot != -1:
                    key = key[:dot]
                group = key.split("/")[0]
                if group in allowed_ids:
                    members_to_keep.append(member)
                    kept += 1

            if members_to_keep:
                tmp_path = out_path + ".tmp"
                with tarfile.open(tmp_path, "w") as tout:
                    tin2 = tarfile.open(input_path, "r")
                    for member in members_to_keep:
                        fobj = tin2.extractfile(member)
                        tout.addfile(member, fobj)
                    tin2.close()
                os.rename(tmp_path, out_path)

        # Mark as done after successful completion
        os.makedirs(os.path.dirname(done_path), exist_ok=True)
        open(done_path, "w").close()

    except Exception as e:
        return (fname, total, kept, str(e))

    return (fname, total, kept, None)


def main():
    parser = argparse.ArgumentParser(description="Filter webdataset shards by allowed archive IDs")
    parser.add_argument("--snapshot-dir", required=True, help="Path to HF hub snapshot directory")
    parser.add_argument("--allowed-ids", required=True, help="Path to allowed archive IDs text file")
    parser.add_argument("--output-dir", required=True, help="Output directory for filtered shards")
    parser.add_argument("--num-workers", type=int, default=288, help="Number of parallel workers")
    args = parser.parse_args()

    # Load allowed IDs
    with open(args.allowed_ids) as f:
        allowed_ids = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(allowed_ids)} allowed IDs")

    # Collect all input shards
    shards = get_all_shard_paths(args.snapshot_dir)
    print(f"Found {len(shards)} input shards")

    # Create output subdirectories
    subdirs = set(s[0] for s in shards)
    for subdir in subdirs:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Process in parallel
    worker_fn = partial(filter_shard, allowed_ids=allowed_ids, output_dir=args.output_dir)

    total_members = 0
    total_kept = 0
    total_shards_written = 0
    failures = []

    with Pool(args.num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_fn, shards),
            total=len(shards),
            desc="Filtering shards",
            unit="shard",
        ):
            fname, n_total, n_kept, error = result
            total_members += n_total
            total_kept += n_kept
            if n_kept > 0:
                total_shards_written += 1
            if error:
                failures.append((fname, error))

    # Summary
    print()
    print("=" * 60)
    print("FILTER SUMMARY")
    print("=" * 60)
    print(f"Input shards:    {len(shards)}")
    print(f"Output shards:   {total_shards_written}")
    print(f"Total samples:   {total_members}")
    print(f"Kept:            {total_kept} ({100*total_kept/total_members:.1f}%)")
    print(f"Removed:         {total_members - total_kept} ({100*(total_members-total_kept)/total_members:.1f}%)")
    print(f"Output dir:      {args.output_dir}")

    if failures:
        print(f"\nFailed shards: {len(failures)}")
        for fname, err in failures:
            print(f"  {fname}: {err}")

    # Clean up .done markers if all shards completed successfully
    done_dir = os.path.join(args.output_dir, ".done")
    if not failures and os.path.exists(done_dir):
        shutil.rmtree(done_dir)
        print("\nCleaned up .done markers")

    print("=" * 60)


if __name__ == "__main__":
    main()
