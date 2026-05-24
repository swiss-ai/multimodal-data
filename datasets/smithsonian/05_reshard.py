import glob
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds

src_path = "/path/to/data/vision-datasets/processed/smithsonian/smithsonian_cleaned4"
dst_path = "/path/to/data/vision-datasets/processed/smithsonian/smithsonian_cleaned5"

os.makedirs(dst_path, exist_ok=True)

tar_files = glob.glob(os.path.join(src_path, "*.tar"))
tar_files.sort()
if not tar_files:
    raise ValueError(f"No .tar files found in {src_path}")

dataset = wds.WebDataset(tar_files, shardshuffle=False)

batch_size = 1000
buffer = []
file_index = 0


def write_batch_to_parquet(data_list, index, output_dir):
    if not data_list:
        return
    df = pd.DataFrame(data_list)
    table = pa.Table.from_pandas(df)
    filename = f"{index:04d}.parquet"
    filepath = os.path.join(output_dir, filename)
    pq.write_table(table, filepath)
    print(f"Wrote {len(data_list)} records to {filename}")


for i, sample in enumerate(dataset):
    ok = True
    for key in ["jpg", "txt", "json"]:
        if key not in sample:
            print(f"Warning: Sample {i} is missing key '{key}'")
            ok = False
            break
    if not ok:
        continue

    buffer.append(
        {
            "id": f"{i:08}___{sample['__key__']}",
            "image": sample["jpg"],
            "caption": sample["txt"].decode("utf-8"),
            "metadata": sample["json"].decode("utf-8"),
        }
    )

    if len(buffer) >= batch_size:
        write_batch_to_parquet(buffer, file_index, dst_path)
        buffer = []
        file_index += 1

if buffer:
    write_batch_to_parquet(buffer, file_index, dst_path)

print("Conversion complete!")
