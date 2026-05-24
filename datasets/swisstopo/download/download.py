import os

import img2dataset

input_file = "/tmp/get_swisstopo_urls/data/swisstopo_urls.csv"
output_folder = "/path/to/data/vision-datasets/swisstopo"
process_count = 1

if __name__ == "__main__":
    os.makedirs(output_folder, exist_ok=True)

    img2dataset.download(
        url_list=input_file,
        output_folder=output_folder,
        image_size=(1024, 1024),
        processes_count=process_count,
        resize_mode="no",
        encode_quality=0,
        encode_format="png",
        skip_reencode=True,
        output_format="webdataset",
        input_format="csv",
        url_col="url",
        save_additional_columns=["pair_id", "image_type", "bbox"],
        number_sample_per_shard=10000,
        timeout=360,
        retries=5,
    )
