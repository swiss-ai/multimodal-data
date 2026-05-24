"""
Merge all task_*.jsonl files from parquet_caption/ into a single parquet.

Run with the dailymed .venv after the recaption job finishes:
    .venv/bin/python merge_captions.py
"""

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

SRC_DIR = os.environ.get(
    "SRC_DIR",
    "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_caption",
)
OUT_FILE = os.environ.get(
    "OUT_FILE",
    "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_caption/captions.parquet",
)

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("doc_id", pa.string()),
        pa.field("image_index", pa.int32()),
        pa.field("image_name", pa.string()),
        pa.field("caption", pa.string()),
    ]
)


def main() -> None:
    jsonl_files = sorted(Path(SRC_DIR).glob("task_*.jsonl"))
    assert jsonl_files, f"No task_*.jsonl in {SRC_DIR}"
    print(f"[merge] {len(jsonl_files)} jsonl files → {OUT_FILE}")

    rows: list[dict] = []
    for jf in tqdm(jsonl_files, unit="file"):
        with open(jf, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    print(f"[merge] {len(rows)} total rows")
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, OUT_FILE, compression="zstd", compression_level=3)
    print(f"[merge] done → {OUT_FILE} ({Path(OUT_FILE).stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
