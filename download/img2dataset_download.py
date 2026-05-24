#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

import img2dataset
import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = Path(os.environ.get("METADATA_DIR", "metadata/blip3_grounding_50m"))
INPUT_DIR = ROOT_DIR / "filtered"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "downloaded"))
CHUNK_SIZE = 320_000
NUMBER_SAMPLE_PER_SHARD = 10_000
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
        processes_count=32,
        thread_count=8,
        resize_mode="no",
        output_format="webdataset",
        input_format="parquet",
        url_col="url",
        caption_col="caption",
        number_sample_per_shard=NUMBER_SAMPLE_PER_SHARD,
        timeout=45,
        retries=2,
        disable_all_reencoding=True,
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

    if not rows:
        final_dir.mkdir(parents=True, exist_ok=True)
        chunk_done_file.write_text("")
        print(f"done {final_dir}")
        return

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
