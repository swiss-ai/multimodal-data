import os
import zipfile

import polars as pl
import webdataset

ROOT_DIR = "/path/to/data/vision-datasets/wangzhecheng"
ZIP_FILES = [
    os.path.join(ROOT_DIR, "SkyScript", "images2.zip"),
    os.path.join(ROOT_DIR, "SkyScript", "images3.zip"),
    os.path.join(ROOT_DIR, "SkyScript", "images4.zip"),
    os.path.join(ROOT_DIR, "SkyScript", "images5.zip"),
    os.path.join(ROOT_DIR, "SkyScript", "images6.zip"),
    os.path.join(ROOT_DIR, "SkyScript", "images7.zip"),
]
CSV_FILE = os.path.join(ROOT_DIR, "SkyScript", "SkyScript_train_unfiltered_5M.csv")

DF = pl.read_csv(CSV_FILE)
map_filepath_to_title = dict()
for row in DF.iter_rows():
    filepath, caption = row[0], row[2]
    map_filepath_to_title[filepath] = caption

pattern = os.path.join(ROOT_DIR, "SkyScript_wds", "part-%06d.tar")
os.makedirs(os.path.dirname(pattern), exist_ok=True)
sink = webdataset.ShardWriter(pattern, maxcount=100000)

for zip_file in ZIP_FILES:
    print(f"Processing {zip_file}...")
    with zipfile.ZipFile(zip_file, "r") as zip_file:
        for name in zip_file.namelist():
            if name.endswith("/"):
                continue
            assert name.endswith(".jpg")

            if name not in map_filepath_to_title:
                print(f"Warning: {name} not found in CSV, skipping.")
                continue

            caption = map_filepath_to_title[name]
            caption = "<|img1|>\n" + caption

            sink.write(
                {
                    "__key__": name[:-4].replace("/", "__"),
                    "img1.jpg": zip_file.read(name),
                    "txt": caption,
                }
            )

sink.close()
