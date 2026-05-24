#!/usr/bin/env python3

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = Path("/tmp/metadata/PD12M/parquet")
OUTPUT_DIR = Path("/tmp/metadata/PD12M/filtered")
BATCH_SIZE = 100_000
OUTPUT_SCHEMA = pa.schema([("url", pa.string()), ("caption", pa.string())])


def copy_file(input_path):
    output_path = OUTPUT_DIR / input_path.name
    temp_path = OUTPUT_DIR / f"{input_path.name}.tmp"
    kept = 0

    if output_path.exists():
        print(f"skip {input_path.name}", flush=True)
        return 0

    if temp_path.exists():
        temp_path.unlink()

    parquet_file = pq.ParquetFile(input_path)
    with pq.ParquetWriter(temp_path, OUTPUT_SCHEMA) as writer:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["url", "caption"]):
            table = pa.table(
                {
                    "url": batch.column("url"),
                    "caption": batch.column("caption"),
                },
                schema=OUTPUT_SCHEMA,
            )
            writer.write_table(table)
            kept += batch.num_rows

    temp_path.rename(output_path)
    print(f"{input_path.name} kept={kept} dropped=0", flush=True)
    return kept


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob("metadata_*.parquet"))
    if not input_files:
        raise RuntimeError(f"No metadata parquet files found in {INPUT_DIR}")

    kept = 0
    for input_path in input_files:
        kept += copy_file(input_path)

    print(f"total kept={kept} dropped=0", flush=True)


if __name__ == "__main__":
    main()
