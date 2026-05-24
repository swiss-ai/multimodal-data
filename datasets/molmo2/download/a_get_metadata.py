#!/usr/bin/env python3

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_DIR = Path(os.environ.get("MOLMO2_DATASET_DIR", ""))
OUTPUT_DIR = Path(os.environ.get("METADATA_OUTPUT_DIR", "/tmp/metadata/Molmo2-MultiImageQA/parquet"))
OUTPUT_PATH = OUTPUT_DIR / "metadata.parquet"
BATCH_SIZE = 4096
OUTPUT_SCHEMA = pa.schema([("url", pa.string())])


def normalize_url(value):
    if value is None:
        return ""
    return str(value).strip()


def iter_image_tables(parquet_paths):
    seen_urls = set()

    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=["image_urls"],
        ):
            data = batch.to_pydict()
            rows = {"url": []}

            for image_urls in data["image_urls"]:
                urls = image_urls or []
                for url in urls:
                    normalized_url = normalize_url(url)
                    if not normalized_url or normalized_url in seen_urls:
                        continue

                    seen_urls.add(normalized_url)
                    rows["url"].append(normalized_url)

            if rows["url"]:
                yield pa.table(rows, schema=OUTPUT_SCHEMA)


def write_metadata(parquet_paths):
    temp_path = OUTPUT_DIR / f"{OUTPUT_PATH.name}.tmp"

    if OUTPUT_PATH.exists():
        print(f"skip {OUTPUT_PATH.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    rows_written = 0
    with pq.ParquetWriter(temp_path, OUTPUT_SCHEMA) as writer:
        for table in iter_image_tables(parquet_paths):
            writer.write_table(table)
            rows_written += table.num_rows

    temp_path.rename(OUTPUT_PATH)
    print(f"{OUTPUT_PATH.name} rows={rows_written}", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(DATASET_DIR.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files found in {DATASET_DIR}")

    write_metadata(parquet_files)


if __name__ == "__main__":
    main()
