import os

import img2dataset

input_file = os.environ.get("URL_FILE", "swisstopo_urls.csv")
output_folder = os.environ.get("OUTPUT_DIR", "./swisstopo_downloaded")

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    img2dataset.download(
        url_list=input_file,
        output_folder=output_folder,
        image_size=512,
        processes_count=1,
        thread_count=256,
        resize_mode="no",
        encode_quality=0,
        encode_format="png",
        output_format="webdataset",
        input_format="csv",
        url_col="url",
        save_additional_columns=["sample_id", "bbox"],
        number_sample_per_shard=10000,
        timeout=120,
        retries=0,
    )
