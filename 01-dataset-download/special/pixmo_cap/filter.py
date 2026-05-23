"""
Filter the PixMo-Cap metadata parquet files, keeping only rows with valid image URLs.

Usage:
    python filter.py --input-dir ./pixmo-cap-meta --output-file filtered.parquet
"""

import argparse
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


def is_valid_image_url(url) -> bool:
    if not url or not isinstance(url, str):
        return False
    parts = urlsplit(url)
    return parts.scheme in ("http", "https") and bool(parts.netloc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the downloaded PixMo-Cap parquet files",
    )
    parser.add_argument(
        "--output-file",
        default="filtered.parquet",
        help="Output parquet file with valid URLs only (default: filtered.parquet)",
    )
    parser.add_argument(
        "--url-col",
        default="image_url",
        help="Name of the URL column (default: image_url)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    parquet_files = sorted(input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    before = len(df)

    df = df[df[args.url_col].map(is_valid_image_url)]
    after = len(df)

    df.to_parquet(args.output_file, index=False)
    print(f"Kept {after:,} / {before:,} rows (dropped {before - after:,} with invalid URLs)")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
