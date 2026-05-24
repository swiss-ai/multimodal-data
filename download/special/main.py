import os

from datasets import load_dataset

columns = [f"image_{i}_path" for i in [1, 2, 3]]

ds = load_dataset(
    "parquet",
    data_dir=os.environ.get("DATA_DIR", "/path/to/data/medical/raw/scin"),
    streaming=True,
).select_columns(columns)["train"]

image_count = 0

print("Starting processing...")
for example in ds:
    for col in columns:
        img = example[col]
        if img is not None:
            image_count += 1

    if image_count % 300 == 0:
        print(f"Processed {image_count} images so far...")

print(f"Total images processed: {image_count}")
