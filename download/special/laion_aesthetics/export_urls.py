"""
Export image URLs from the dclure/laion-aesthetics-12m-umap HF dataset.

The HF dataset contains metadata (URLs, scores, UMAP coordinates) but not images.
Run this script first, then use img2dataset to download the actual images.

Usage:
    python export_urls.py --output-file all_urls.parquet
"""

import argparse

import pandas as pd

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-file",
        default="all_urls.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Batch size for streaming iteration",
    )
    args = parser.parse_args()

    print("Loading dataset in streaming mode...")
    ds = load_dataset("dclure/laion-aesthetics-12m-umap", split="train", streaming=True)

    records = []
    for batch in ds.iter(batch_size=args.batch_size):
        urls = batch.get("URL") or batch.get("url") or []
        records.extend({"url": u} for u in urls if u)

    df = pd.DataFrame(records)
    df.to_parquet(args.output_file, index=False)
    print(f"Saved {len(df):,} URLs to {args.output_file}")


if __name__ == "__main__":
    main()
