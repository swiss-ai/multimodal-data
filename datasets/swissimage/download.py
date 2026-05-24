import os

import img2dataset

input_file = os.environ.get("SWISSIMAGE_URLS", "/tmp/downloads/vids/swissimage/swissimage.csv")
output_folder = os.environ.get("SWISSIMAGE_OUTPUT_DIR", "")
process_count = 8

if __name__ == "__main__":
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    os.makedirs(output_folder, exist_ok=True)

    img2dataset.download(
        url_list=input_file,
        output_folder=output_folder,
        processes_count=process_count,
        resize_mode="no",
        output_format="webdataset",
        input_format="txt",
        number_sample_per_shard=100,
        timeout=120,
        retries=5,
    )
