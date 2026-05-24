#!/usr/bin/env python3

import argparse
from pathlib import Path

import img2dataset

ROOT_DIR = Path("/tmp/metadata/megalith_10m_florence2")
INPUT_DIR = ROOT_DIR / "filtered"
OUTPUT_DIR = Path("/path/to/data/vision-datasets/hf___aipicasso___megalith-10m-florence2___downloaded")
PROCESSES_COUNT = 1
THREAD_COUNT = 16
IMAGE_SIZE = 4000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-id", required=True)
    return parser.parse_args()


def download_part(input_parquet, output_dir):
    img2dataset.download(
        url_list=str(input_parquet),
        output_folder=str(output_dir),
        processes_count=PROCESSES_COUNT,
        thread_count=THREAD_COUNT,
        image_size=IMAGE_SIZE,
        resize_mode="keep_ratio_largest",
        resize_only_if_bigger=True,
        output_format="webdataset",
        input_format="parquet",
        url_col="url_highres",
        save_additional_columns=["url_source"],
        caption_col="caption",
        number_sample_per_shard=10000,
        timeout=90,
        retries=0,
        disable_all_reencoding=True,
        incremental_mode="incremental",
    )


def process_part(part_id):
    input_parquet = INPUT_DIR / f"metadata_{part_id}.parquet"
    if not input_parquet.exists():
        print(f"missing {input_parquet}, skipping")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    final_dir = OUTPUT_DIR / part_id
    done_file = final_dir / "_SUCCESS"
    if done_file.exists():
        print(f"already done {part_id}")
        return

    if final_dir.exists():
        print(f"resume existing {final_dir}")

    download_part(input_parquet, final_dir)
    (final_dir / "_SUCCESS").write_text("")
    print(f"finished {part_id}")


def main():
    args = parse_args()
    process_part(args.part_id)


if __name__ == "__main__":
    main()
