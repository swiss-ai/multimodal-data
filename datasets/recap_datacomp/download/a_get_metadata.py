#!/usr/bin/env python3

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_DIR = Path(os.environ.get("RECAP_DATACOMP_DIR", ""))
OUTPUT_DIR = Path(os.environ.get("METADATA_OUTPUT_DIR", "/tmp/metadata/Recap-DataComp-1B/parquet"))
BATCH_SIZE = 10000
OUTPUT_SCHEMA = pa.schema([("url", pa.string()), ("caption", pa.string())])


def iter_batches(parquet_path):
    parquet_file = pq.ParquetFile(parquet_path)
    yield from parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["url", "re_caption"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(DATASET_DIR.glob("train-*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {DATASET_DIR}")

    for parquet_path in parquet_files:
        part_id = parquet_path.name.split("-")[1]
        output_path = OUTPUT_DIR / f"metadata_{part_id}.parquet"

        rows_written = 0
        with pq.ParquetWriter(output_path, OUTPUT_SCHEMA) as writer:
            for batch in iter_batches(parquet_path):
                table = pa.table(
                    {
                        "url": batch.column("url"),
                        "caption": batch.column("re_caption"),
                    },
                    schema=OUTPUT_SCHEMA,
                )
                writer.write_table(table)
                rows_written += batch.num_rows

        print(f"{output_path} {rows_written}")


if __name__ == "__main__":
    main()
