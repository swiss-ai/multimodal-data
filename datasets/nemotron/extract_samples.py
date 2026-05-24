#!/usr/bin/env python3
"""Extract 20 samples from each mirrored dataset using DuckDB."""

from pathlib import Path

import duckdb
import orjson as json

MIRROR_BASE = Path(
    "/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/swissai___Nemotron-Image-Training-v3"
)
OUTPUT_BASE = Path("/tmp/samples/nemotron")
N_SAMPLES = 20


def extract_dataset(con, dataset_name):
    out_dir = OUTPUT_BASE / dataset_name
    if out_dir.exists() and len(list(out_dir.iterdir())) >= N_SAMPLES:
        return

    mirror_dir = MIRROR_BASE / dataset_name
    parquet_files = sorted(mirror_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"[{dataset_name}] No parquet files, skipping")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    glob_pattern = str(mirror_dir / "*.parquet")
    rows = con.sql(
        f"SELECT id, messages, images FROM read_parquet('{glob_pattern}') USING SAMPLE {N_SAMPLES}"
    ).fetchall()

    for sid, msg, img_list in rows:
        sample_dir = out_dir / sid
        if sample_dir.exists():
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "messages.json").write_bytes(json.dumps(json.loads(msg)))
        for key, data in img_list.items() if isinstance(img_list, dict) else img_list:
            img_path = sample_dir / key
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(data)

    print(f"[{dataset_name}] {len(rows)} samples")


def main():
    con = duckdb.connect()
    datasets = sorted(d.name for d in MIRROR_BASE.iterdir() if d.is_dir())
    for ds in datasets:
        extract_dataset(con, ds)


if __name__ == "__main__":
    main()
