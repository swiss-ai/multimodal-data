import os

import img2dataset

output_folder = os.environ.get("OUTPUT_FOLDER", "/tmp/pixmo-cap-dl")

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)
    img2dataset.download(
        url_list="filtered.parquet",
        output_folder=output_folder,
        processes_count=75,
        thread_count=64,
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
