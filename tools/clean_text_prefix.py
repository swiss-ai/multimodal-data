#!/usr/bin/env python3
"""
Clean dataset by copying it to a ___cleaned suffix directory,
removing the first N lines from every .txt file.

Usage:
    python clean_text_prefix.py <path> <num_lines_to_remove>
"""

import argparse
import io
import os
import shutil
import sys
import tarfile
from multiprocessing import Pool, cpu_count
from pathlib import Path


def process_tar_file(args):
    """Process a single tar file: copy all members, removing N lines from .txt files."""
    src_tar_path, dst_dir, num_lines = args
    dst_tar_path = os.path.join(dst_dir, os.path.basename(src_tar_path))

    with tarfile.open(src_tar_path, "r") as src_tar:
        with tarfile.open(dst_tar_path, "w") as dst_tar:
            for member in src_tar.getmembers():
                f = src_tar.extractfile(member)
                if f is None:
                    dst_tar.addfile(member)
                    continue

                data = f.read()

                if member.name.endswith(".txt"):
                    # Decode, remove first N lines, re-encode
                    try:
                        text = data.decode("utf-8")
                    except UnicodeDecodeError:
                        text = data.decode("utf-8", errors="replace")

                    lines = text.splitlines()
                    if len(lines) > num_lines:
                        lines = lines[num_lines:]
                    else:
                        lines = []
                    new_text = "\n".join(lines)
                    if text.endswith("\n") or (lines and lines[-1] == ""):
                        # Preserve trailing newline if original had one
                        if not new_text.endswith("\n"):
                            new_text += "\n"

                    data = new_text.encode("utf-8")
                    member.size = len(data)

                dst_tar.addfile(member, io.BytesIO(data))

    return src_tar_path


def process_sample_dir(src_sample_dir, dst_sample_dir, num_lines):
    """Copy sample directory, removing N lines from .txt files."""
    os.makedirs(dst_sample_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_sample_dir):
        rel_root = os.path.relpath(root, src_sample_dir)
        dst_root = os.path.join(dst_sample_dir, rel_root)
        os.makedirs(dst_root, exist_ok=True)

        for d in dirs:
            os.makedirs(os.path.join(dst_root, d), exist_ok=True)

        for fname in files:
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(dst_root, fname)

            if fname.endswith(".txt"):
                with open(src_path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
                if len(lines) > num_lines:
                    lines = lines[num_lines:]
                else:
                    lines = []
                content = "\n".join(lines)
                # Check if original ended with newline
                with open(src_path, "rb") as f:
                    raw = f.read()
                if raw.endswith(b"\n"):
                    if not content.endswith("\n"):
                        content += "\n"
                with open(dst_path, "w", encoding="utf-8") as f:
                    f.write(content)
            else:
                shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Clean dataset by removing first N lines from .txt files")
    parser.add_argument("path", help="Source dataset directory path")
    parser.add_argument(
        "num_lines",
        type=int,
        help="Number of lines to remove from the start of each .txt file",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() // 2),
        help="Number of parallel workers for tar processing",
    )
    args = parser.parse_args()

    src_dir = Path(args.path).resolve()
    if not src_dir.exists():
        print(f"Error: source path does not exist: {src_dir}", file=sys.stderr)
        sys.exit(1)

    dst_dir = Path(str(src_dir) + "___cleaned")
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"Removing first {args.num_lines} line(s) from .txt files")
    print(f"Workers: {args.workers}")
    print()

    # Collect tar files
    tar_files = sorted(src_dir.glob("part-*.tar"))
    print(f"Found {len(tar_files)} tar file(s) to process")

    # Process tar files in parallel
    if tar_files:
        pool_args = [(str(t), str(dst_dir), args.num_lines) for t in tar_files]

        with Pool(processes=args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_process_tar_wrapper, pool_args), 1):
                print(f"  [{i}/{len(tar_files)}] Done", flush=True)

    # Copy non-tar, non-sample files as-is
    for item in src_dir.iterdir():
        if item.name == "sample":
            continue
        if item.name.startswith("part-") and item.suffix == ".tar":
            continue
        dst_item = dst_dir / item.name
        if item.is_dir():
            if dst_item.exists():
                shutil.rmtree(dst_item)
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)
        print(f"  Copied: {item.name}")

    # Process sample directory if it exists
    src_sample = src_dir / "sample"
    if src_sample.exists():
        dst_sample = dst_dir / "sample"
        print("\nProcessing sample directory...")
        process_sample_dir(str(src_sample), str(dst_sample), args.num_lines)
        print("  Sample directory done")

    print(f"\nDone. Cleaned dataset at: {dst_dir}")


def _process_tar_wrapper(args):
    """Wrapper for multiprocessing pool."""
    return process_tar_file(args)


if __name__ == "__main__":
    main()
