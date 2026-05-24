import os

import img2dataset

output_folder = os.environ.get(
    "OUTPUT_DIR",
    "/path/to/data/vision-datasets/ign_city_tiles",
)
input_file = os.environ.get("INPUT_CSV", "france_tiles_template.csv")
process_count = int(os.environ.get("PROCESS_COUNT", "1"))

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    img2dataset.download(
        url_list=input_file,
        output_folder=output_folder,
        image_size=(512, 512),
        processes_count=process_count,
        resize_mode="no",
        encode_quality=0,
        encode_format="png",
        skip_reencode=True,
        output_format="webdataset",
        input_format="csv",
        url_col="url",
        number_sample_per_shard=1000,
        timeout=120,
        retries=5,
        thread_count=4,
    )
