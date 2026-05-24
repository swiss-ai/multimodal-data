import argparse

import img2dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--process_count", type=int, required=True)
    args = parser.parse_args()

    img2dataset.download(
        url_list=args.input_file,
        output_folder=args.output_folder,
        processes_count=args.process_count,
        resize_mode="no",
        output_format="webdataset",
        input_format="txt",
        number_sample_per_shard=20_000,
        min_image_size=256,
        timeout=30,
        retries=3,
    )
