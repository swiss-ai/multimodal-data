#!/usr/bin/env python3

from pathlib import Path

import img2dataset

INPUT_PATH = Path("/tmp/metadata/Molmo2-MultiImageQA/filtered/metadata.parquet")
OUTPUT_DIR = Path("/path/to/data/vision-datasets/processed/hf___allenai___Molmo2-MultiImageQA___downloaded")


def main():
    if not INPUT_PATH.exists():
        raise RuntimeError(f"Missing filtered parquet: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    img2dataset.download(
        url_list=str(INPUT_PATH),
        # input
        input_format="parquet",
        url_col="url",
        # output
        output_folder=str(OUTPUT_DIR),
        output_format="parquet",
        number_sample_per_shard=10_000,
        oom_shard_count=4,
        # perf
        processes_count=32,
        thread_count=32,
        timeout=60,
        retries=1,
        # encode
        resize_mode="no",
        encode_quality=100,
        encode_format="jpg",
        min_image_size=32,
        max_image_area=100_000_000,
    )


if __name__ == "__main__":
    main()
