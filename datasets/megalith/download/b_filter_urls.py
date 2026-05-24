#!/usr/bin/env python3

import shutil
from pathlib import Path

INPUT_DIR = Path("/tmp/metadata/megalith_10m_florence2/parquet")
OUTPUT_DIR = Path("/tmp/metadata/megalith_10m_florence2/filtered")


def copy_file(input_path):
    output_path = OUTPUT_DIR / input_path.name
    temp_path = OUTPUT_DIR / f"{input_path.name}.tmp"

    if output_path.exists():
        print(f"skip {input_path.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    shutil.copyfile(input_path, temp_path)
    temp_path.rename(output_path)
    print(f"copied {input_path.name}", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob("metadata_*.parquet"))
    if not input_files:
        raise RuntimeError(f"No metadata parquet files found in {INPUT_DIR}")

    copied = 0
    skipped = 0

    for input_path in input_files:
        output_path = OUTPUT_DIR / input_path.name
        if output_path.exists():
            print(f"skip {input_path.name}", flush=True)
            skipped += 1
            continue

        copy_file(input_path)
        copied += 1

    print(f"total copied={copied} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
