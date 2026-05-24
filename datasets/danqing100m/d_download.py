#!/usr/bin/env python3

import argparse
import csv
import shutil
from pathlib import Path

import img2dataset

ROOT_DIR = Path("/tmp/metadata/DanQing100M")
INPUT_DIR = ROOT_DIR / "verified"
OUTPUT_DIR = Path("/path/to/data/vision-datasets/hf___DeepGlint-AI___DanQing100M_images")
CHUNK_SIZE = 10_000
PROCESSES_COUNT = 1
THREAD_COUNT = 256
NUMBER_SAMPLE_PER_SHARD = 10_000
TIMEOUT = 30
RETRIES = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-id", required=True)
    return parser.parse_args()


def write_chunk_csv(chunk_csv, rows):
    chunk_csv.parent.mkdir(parents=True, exist_ok=True)
    with chunk_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def download_chunk(chunk_csv, tmp_dir):
    img2dataset.download(
        url_list=str(chunk_csv),
        output_folder=str(tmp_dir),
        processes_count=PROCESSES_COUNT,
        thread_count=THREAD_COUNT,
        resize_mode="no",
        output_format="webdataset",
        input_format="csv",
        url_col="url",
        caption_col="caption",
        number_sample_per_shard=NUMBER_SAMPLE_PER_SHARD,
        timeout=TIMEOUT,
        retries=RETRIES,
        disable_all_reencoding=True,
    )


def process_chunk(part_dir, chunk_index, rows):
    final_dir = part_dir / f"chunk_{chunk_index:02d}"
    if final_dir.exists():
        print(f"skip {final_dir}")
        return

    tmp_dir = part_dir / f"chunk_{chunk_index:02d}.tmp"
    chunk_csv = part_dir / "csv" / f"chunk_{chunk_index:02d}.csv"

    write_chunk_csv(chunk_csv, rows)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    download_chunk(chunk_csv, tmp_dir)
    (tmp_dir / "_SUCCESS").write_text("")
    tmp_dir.rename(final_dir)
    print(f"done {final_dir}")


def process_part(part_id):
    input_csv = INPUT_DIR / f"metadata_{part_id}.csv"
    if not input_csv.exists():
        print(f"missing {input_csv}, skipping")
        return

    part_dir = OUTPUT_DIR / part_id
    done_file = part_dir / "_SUCCESS"
    if done_file.exists():
        print(f"already done {part_id}")
        return

    part_dir.mkdir(parents=True, exist_ok=True)

    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        chunk_index = 0

        for row in reader:
            rows.append({"url": row["url"], "caption": row["caption"]})
            if len(rows) == CHUNK_SIZE:
                process_chunk(part_dir, chunk_index, rows)
                rows = []
                chunk_index += 1

        if rows:
            process_chunk(part_dir, chunk_index, rows)

    done_file.write_text("")
    print(f"finished {part_id}")


def main():
    args = parse_args()
    process_part(args.part_id)


if __name__ == "__main__":
    main()
