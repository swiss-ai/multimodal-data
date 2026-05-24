#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

import img2dataset
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = Path("/tmp/metadata/FaceCaption-15M")
INPUT_DIR = ROOT_DIR / "filtered"
OUTPUT_DIR = Path("/path/to/data/vision-datasets/raw/OpenFace-CQUPT___FaceCaption-15M___downloaded")
CHUNK_SIZE = 160_000
PROCESSES_COUNT = 16
THREAD_COUNT = 8
NUMBER_SAMPLE_PER_SHARD = 10_000
TIMEOUT = 15
RETRIES = 0
MIN_IMAGE_SIZE = 64
MAX_IMAGE_AREA = 1024 * 1024 * 16
INPUT_SCHEMA = pa.schema(
    [
        ("_id", pa.string()),
        ("url", pa.string()),
        ("caption", pa.string()),
        ("laion_caption", pa.string()),
        ("box", pa.string()),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-id", required=True)
    return parser.parse_args()


def write_chunk_parquet(chunk_parquet, rows):
    chunk_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "_id": [row["_id"] for row in rows],
            "url": [row["url"] for row in rows],
            "caption": [row["caption"] for row in rows],
            "laion_caption": [row["laion_caption"] for row in rows],
            "box": [row["box"] for row in rows],
        },
        schema=INPUT_SCHEMA,
    )
    tmp_parquet = chunk_parquet.with_name(f"{chunk_parquet.name}.tmp.{os.getpid()}")
    pq.write_table(table, tmp_parquet)
    tmp_parquet.replace(chunk_parquet)


def download_chunk(chunk_parquet, tmp_dir):
    img2dataset.download(
        url_list=str(chunk_parquet),
        output_folder=str(tmp_dir),
        processes_count=PROCESSES_COUNT,
        thread_count=THREAD_COUNT,
        output_format="webdataset",
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        save_additional_columns=["_id", "laion_caption", "box"],
        number_sample_per_shard=NUMBER_SAMPLE_PER_SHARD,
        timeout=TIMEOUT,
        retries=RETRIES,
        resize_mode="no",
        min_image_size=MIN_IMAGE_SIZE,
        max_image_area=MAX_IMAGE_AREA,
        encode_format="jpg",
        encode_quality=100,
    )


def cleanup_stale_chunk_dirs(part_dir):
    for tmp_dir in sorted(part_dir.glob("chunk_*.tmp")):
        final_dir = part_dir / tmp_dir.stem
        if (tmp_dir / "_SUCCESS").exists() and not final_dir.exists():
            print(f"promote completed tmp {tmp_dir}")
            tmp_dir.rename(final_dir)
            continue

        print(f"remove stale {tmp_dir}")
        shutil.rmtree(tmp_dir)

    for final_dir in sorted(part_dir.glob("chunk_*")):
        if not final_dir.is_dir() or final_dir.suffix == ".tmp":
            continue
        if not (final_dir / "_SUCCESS").exists():
            print(f"remove stale {final_dir}")
            shutil.rmtree(final_dir)


def process_chunk(part_dir, chunk_index, rows):
    final_dir = part_dir / f"chunk_{chunk_index:03d}"
    chunk_done_file = final_dir / "_SUCCESS"
    if chunk_done_file.exists():
        print(f"skip {final_dir}")
        return

    tmp_dir = part_dir / f"chunk_{chunk_index:03d}.tmp"
    chunk_parquet = part_dir / "parquet" / f"chunk_{chunk_index:03d}.parquet"

    if final_dir.exists():
        print(f"remove stale {final_dir}")
        shutil.rmtree(final_dir)

    write_chunk_parquet(chunk_parquet, rows)

    if tmp_dir.exists():
        print(f"remove stale {tmp_dir}")
        shutil.rmtree(tmp_dir)

    download_chunk(chunk_parquet, tmp_dir)
    (tmp_dir / "_SUCCESS").write_text("")
    tmp_dir.rename(final_dir)
    print(f"done {final_dir}")


def process_part(part_id):
    input_parquet = INPUT_DIR / f"metadata_{part_id}.parquet"
    if not input_parquet.exists():
        print(f"missing {input_parquet}, skipping")
        return

    part_dir = OUTPUT_DIR / part_id
    done_file = part_dir / "_SUCCESS"
    if done_file.exists():
        print(f"already done {part_id}")
        return

    part_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_chunk_dirs(part_dir)

    parquet_file = pq.ParquetFile(input_parquet)
    rows = []
    chunk_index = 0

    for batch in parquet_file.iter_batches(
        batch_size=CHUNK_SIZE,
        columns=["_id", "url", "caption", "laion_caption", "box"],
    ):
        data = batch.to_pydict()
        for row_id, url, caption, laion_caption, box in zip(
            data["_id"],
            data["url"],
            data["caption"],
            data["laion_caption"],
            data["box"],
        ):
            rows.append(
                {
                    "_id": row_id,
                    "url": url,
                    "caption": caption,
                    "laion_caption": laion_caption,
                    "box": box,
                }
            )
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
