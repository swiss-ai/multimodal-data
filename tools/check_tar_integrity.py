#!/usr/bin/env python3

import argparse
import os
import tarfile
import time

DEFAULT_ROOT = "/path/to/data/vision-datasets/hf___Salesforce___blip3-grounding-50m___downloaded"
DEFAULT_PROGRESS_SECS = 60
READ_CHUNK_SIZE = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--progress-secs", type=int, default=DEFAULT_PROGRESS_SECS)
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based tar index to start from after sorting and exclusions.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="1-based tar index to stop at, inclusive.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=["_tmp"],
        help="Directory name to skip while walking the tree. Can be passed multiple times.",
    )
    return parser.parse_args()


def iter_tar_paths(root, exclude_dirnames):
    exclude_dirnames = set(exclude_dirnames)
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in exclude_dirnames)
        for filename in sorted(filenames):
            if filename.endswith(".tar"):
                yield os.path.join(current_root, filename)


def drain_fileobj(fileobj):
    while True:
        chunk = fileobj.read(READ_CHUNK_SIZE)
        if not chunk:
            return


def check_tar(archive_path):
    member_count = 0
    byte_count = 0

    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar:
            member_count += 1
            if not member.isfile():
                continue

            byte_count += member.size
            fileobj = tar.extractfile(member)
            if fileobj is None:
                raise OSError(f"Could not read member payload: {member.name}")

            with fileobj:
                drain_fileobj(fileobj)

    return member_count, byte_count


def format_bytes(num_bytes):
    gib = 1024**3
    tib = 1024**4
    if num_bytes >= tib:
        return f"{num_bytes / tib:.2f} TiB"
    return f"{num_bytes / gib:.2f} GiB"


def main():
    args = parse_args()
    all_tar_paths = list(iter_tar_paths(args.root, args.exclude_dir))
    total = len(all_tar_paths)
    start_index = max(args.start_index, 1)
    end_index = total if args.end_index is None else min(args.end_index, total)

    if start_index > total:
        print(f"Requested start index {start_index} is beyond total tar count {total}")
        return
    if end_index < start_index:
        print(f"Invalid range: start_index={start_index}, end_index={end_index}")
        return

    tar_paths = all_tar_paths[start_index - 1 : end_index]
    slice_total = len(tar_paths)

    print(f"Checking tar files {start_index}-{end_index} of {total} under {args.root}")
    if args.exclude_dir:
        print(f"Skipping directories named: {', '.join(sorted(set(args.exclude_dir)))}")

    checked = 0
    ok = 0
    bad = 0
    total_members = 0
    total_bytes = 0
    failures = []
    start_time = time.time()
    last_report = start_time

    for archive_path in tar_paths:
        checked += 1
        try:
            member_count, byte_count = check_tar(archive_path)
        except (tarfile.TarError, OSError, EOFError) as exc:
            bad += 1
            failures.append((archive_path, str(exc)))
            print(f"BAD\t{archive_path}\t{exc}", flush=True)
        else:
            ok += 1
            total_members += member_count
            total_bytes += byte_count

        now = time.time()
        if now - last_report >= args.progress_secs:
            elapsed = now - start_time
            rate = checked / elapsed if elapsed else 0.0
            print(
                "PROGRESS\t"
                f"checked={checked}/{slice_total}\t"
                f"ok={ok}\tbad={bad}\t"
                f"members={total_members}\t"
                f"payload={format_bytes(total_bytes)}\t"
                f"rate={rate:.2f} tar/s",
                flush=True,
            )
            last_report = now

    elapsed = time.time() - start_time
    print(
        "DONE\t"
        f"checked={checked}/{slice_total}\t"
        f"ok={ok}\tbad={bad}\t"
        f"members={total_members}\t"
        f"payload={format_bytes(total_bytes)}\t"
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )

    if failures:
        print("FAILURES", flush=True)
        for archive_path, error in failures:
            print(f"{archive_path}\t{error}", flush=True)


if __name__ == "__main__":
    main()
