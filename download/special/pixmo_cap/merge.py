import os

import polars as pl

parquet_dir = "/path/to/data/vision-datasets/pixmo-cap/data"
parquet_glob = os.path.join(parquet_dir, "*.parquet")
df = pl.read_parquet(parquet_glob)
df.write_parquet("data.parquet")
