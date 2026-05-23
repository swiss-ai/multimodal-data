"""
Download PixMo-Cap images using img2dataset.

Run filter.py first to produce filtered.parquet, then:

    python download.py \
        --input-file filtered.parquet \
        --output-dir /path/to/pixmo-cap-images

Requirements:
    pip install img2dataset
"""

import argparse
import os

import img2dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        default="filtered.parquet",
        help="Input parquet file with image URLs (default: filtered.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for downloaded images",
    )
    parser.add_argument(
        "--processes-count",
        type=int,
        default=32,
        help="Number of download processes (default: 32)",
    )
    parser.add_argument(
        "--thread-count",
        type=int,
        default=64,
        help="Number of threads per process (default: 64)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img2dataset.download(
        url_list=args.input_file,
        output_folder=args.output_dir,
        processes_count=args.processes_count,
        thread_count=args.thread_count,
        resize_mode="no",
        encode_format="png",
        encode_quality=0,
        output_format="webdataset",
        input_format="parquet",
        url_col="image_url",
        caption_col="caption",
        save_additional_columns=["transcripts"],
        number_sample_per_shard=10000,
        timeout=30,
        retries=1,
    )


if __name__ == "__main__":
    main()
