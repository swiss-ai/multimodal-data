"""img2dataset wrapper for downloading the WMS URLs into a webdataset.

Usage:
    python download.py --urls data/sample_urls.csv \
        --out data/sample_wds --processes 2 --threads 64 \
        --shard-size 1000
"""

import argparse
import shutil
from pathlib import Path

from img2dataset import download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--clean", action="store_true", help="Delete output dir before running")
    args = ap.parse_args()

    out = Path(args.out)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    download(
        url_list=args.urls,
        output_folder=str(out),
        processes_count=args.processes,
        thread_count=args.threads,
        resize_mode="no",
        encode_format="png",
        encode_quality=0,  # no PNG recompression — fastest, ~2.6 MB/img
        output_format="webdataset",
        input_format="csv",
        url_col="url",
        save_additional_columns=[
            "sample_id",
            "tile_id",
            "bbox",
            "layer",
            "img_w",
            "img_h",
            "lang",
            "scale",
            "building_frac",
        ],
        number_sample_per_shard=args.shard_size,
        timeout=120,
        retries=1,
    )


if __name__ == "__main__":
    main()
