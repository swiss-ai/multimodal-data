import argparse
import os

import img2dataset


def main():
    parser = argparse.ArgumentParser(description="Download SWISSIMAGE tiles via img2dataset")
    parser.add_argument("--input-file", required=True, help="CSV/TXT with image URLs")
    parser.add_argument("--output-folder", required=True, help="Output directory for WDS shards")
    parser.add_argument("--process-count", type=int, default=8, help="Number of download processes")
    args = parser.parse_args()

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.makedirs(args.output_folder, exist_ok=True)

    img2dataset.download(
        url_list=args.input_file,
        output_folder=args.output_folder,
        processes_count=args.process_count,
        resize_mode="no",
        output_format="webdataset",
        input_format="txt",
        number_sample_per_shard=100,
        timeout=120,
        retries=5,
    )


if __name__ == "__main__":
    main()
