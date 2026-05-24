#!/usr/bin/env python3

import csv
from pathlib import Path

import pyarrow.parquet as pq

DATASET_DIR = Path("/path/to/data/vision-datasets/hf___DeepGlint-AI___DanQing100M/data")
OUTPUT_DIR = Path("/tmp/metadata/DanQing100M/csv")
TOTAL_ROWS = 99_892_381
ROWS_PER_FILE = 100_000
NUM_FILES = (TOTAL_ROWS + ROWS_PER_FILE - 1) // ROWS_PER_FILE
BATCH_SIZE = 8192


def iter_pairs():
    for parquet_path in sorted(DATASET_DIR.glob("*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["url", "recaption"]):
            data = batch.to_pydict()
            for url, caption in zip(data["url"], data["recaption"]):
                yield url, caption


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = iter_pairs()
    rows_written = 0

    for file_index in range(NUM_FILES):
        rows_in_file = min(ROWS_PER_FILE, TOTAL_ROWS - rows_written)
        output_path = OUTPUT_DIR / f"metadata_{file_index:04d}.csv"

        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["url", "caption"])

            for _ in range(rows_in_file):
                url, caption = next(pairs)
                writer.writerow([url, caption])
                rows_written += 1

        print(f"{output_path} {rows_in_file}")

    if rows_written != TOTAL_ROWS:
        raise RuntimeError(f"Expected {TOTAL_ROWS} rows, wrote {rows_written}")


if __name__ == "__main__":
    main()
