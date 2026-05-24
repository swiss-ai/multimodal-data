import os
import pickle

import polars as pl

root_dir = "/path/to/data/vision-datasets/hf_hub_cache/datasets--imageomics--TreeOfLife-10M/snapshots/91debffb7146c32c89d76feb1eb575b555e2ecc7"
license_path = os.path.join(root_dir, "metadata/licenses.csv")

df = pl.read_csv(
    license_path,
    has_header=True,
    columns=["treeoflife_id", "license_name"],
)
mapping = dict(
    zip(
        df["treeoflife_id"].to_list(),
        df["license_name"].to_list(),
    )
)

assert len(mapping) == len(df)
print(f"Total licenses: {len(df)}")
print(f"Unique licenses: {len(df['license_name'].unique())}")

license_counts = df.group_by("license_name").len().sort("license_name")
with pl.Config(tbl_rows=100):
    print(license_counts)

os.makedirs("data", exist_ok=True)
with open("data/license_dict.pkl", "wb") as f:
    pickle.dump(mapping, f)

with open("data/license_dict.pkl", "rb") as f:
    loaded_mapping = pickle.load(f)
assert mapping == loaded_mapping
