#!/usr/bin/env python3

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_DIR = Path("/path/to/data/vision-datasets/hf___Salesforce___blip3-grounding-50m/data")
OUTPUT_DIR = Path("/tmp/metadata/blip3_grounding_50m/parquet")
BATCH_SIZE = 16384
OUTPUT_SCHEMA = pa.schema([("url", pa.string()), ("caption", pa.string())])


def iter_batches(parquet_path):
    parquet_file = pq.ParquetFile(parquet_path)
    yield from parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["url", "cogvlm_caption"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(DATASET_DIR.glob("combined_part_*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {DATASET_DIR}")

    for parquet_path in parquet_files:
        part_id = parquet_path.stem.rsplit("_", 1)[-1]
        output_path = OUTPUT_DIR / f"metadata_{part_id}.parquet"

        rows_written = 0
        with pq.ParquetWriter(output_path, OUTPUT_SCHEMA) as writer:
            for batch in iter_batches(parquet_path):
                table = pa.table(
                    {
                        "url": batch.column("url"),
                        "caption": batch.column("cogvlm_caption"),
                    },
                    schema=OUTPUT_SCHEMA,
                )
                writer.write_table(table)
                rows_written += batch.num_rows

        print(f"{output_path} {rows_written}")


if __name__ == "__main__":
    main()
