#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

import img2dataset
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = Path(os.environ.get("PD12M_ROOT", "/tmp/metadata/PD12M"))
INPUT_DIR = ROOT_DIR / "filtered"
OUTPUT_DIR = Path(os.environ.get("PD12M_DOWNLOAD_DIR", ""))
CHUNK_SIZE = 320_000
PROCESSES_COUNT = 16
THREAD_COUNT = 8
NUMBER_SAMPLE_PER_SHARD = 10_000
TIMEOUT = 30
RETRIES = 3
INPUT_SCHEMA = pa.schema([("url", pa.string()), ("caption", pa.string())])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-id", required=True)
    return parser.parse_args()


def write_chunk_parquet(chunk_parquet, rows):
    chunk_parquet.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "url": [row["url"] for row in rows],
            "caption": [row["caption"] for row in rows],
        },
        schema=INPUT_SCHEMA,
    )
    pq.write_table(table, chunk_parquet)


def download_chunk(chunk_parquet, tmp_dir):
    img2dataset.download(
        url_list=str(chunk_parquet),
        output_folder=str(tmp_dir),
        processes_count=PROCESSES_COUNT,
        thread_count=THREAD_COUNT,
        resize_mode="no",
        output_format="webdataset",
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        number_sample_per_shard=NUMBER_SAMPLE_PER_SHARD,
        timeout=TIMEOUT,
        retries=RETRIES,
        disable_all_reencoding=True,
    )


def process_chunk(part_dir, chunk_index, rows):
    final_dir = part_dir / f"chunk_{chunk_index:03d}"
    if final_dir.exists():
        print(f"skip {final_dir}")
        return

    tmp_dir = part_dir / f"chunk_{chunk_index:03d}.tmp"
    chunk_parquet = part_dir / "parquet" / f"chunk_{chunk_index:03d}.parquet"

    write_chunk_parquet(chunk_parquet, rows)

    if tmp_dir.exists():
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

    parquet_file = pq.ParquetFile(input_parquet)
    rows = []
    chunk_index = 0

    for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=["url", "caption"]):
        data = batch.to_pydict()
        for url, caption in zip(data["url"], data["caption"]):
            rows.append({"url": url, "caption": caption})
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
