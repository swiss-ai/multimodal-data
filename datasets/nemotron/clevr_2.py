"""Convert clevr_2 dataset: JSONL refs CLEVR_v1.0 images (same zip as clevr_1)."""

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path("/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3")
JSONL = BASE / "hf___nvidia___Nemotron-Image-Training-v3/clevr_2/clevr_2.jsonl"
ZIP = BASE / "datasets/clevr_1/CLEVR_v1.0.zip"  # same source as clevr_1
OUT_DIR = BASE / "swissai___Nemotron-Image-Training-v3/clevr_2"
OUT_FILE = OUT_DIR / "clevr_2.parquet"

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("messages", pa.string()),
        ("images", pa.map_(pa.string(), pa.binary())),
    ]
)
BATCH_SIZE = 500


def load_jsonl():
    img_to_rows = defaultdict(list)
    with open(JSONL) as f:
        for line in f:
            row = json.loads(line)
            for msg in row["messages"]:
                for c in msg["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        img_to_rows[c["image"]].append((row["id"], json.dumps(row["messages"])))
    print(f"Loaded {sum(len(v) for v in img_to_rows.values())} JSONL entries covering {len(img_to_rows)} unique images")
    return img_to_rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading JSONL into memory...")
    img_to_rows = load_jsonl()

    ids, messages, images = [], [], []
    matched = 0
    skipped = 0

    print(f"Opening zip {ZIP} and iterating images...")
    with zipfile.ZipFile(ZIP) as zf, pq.ParquetWriter(OUT_FILE, SCHEMA) as writer:
        for info in zf.infolist():
            if not info.filename.endswith(".png"):
                continue
            basename = Path(info.filename).name
            rows = img_to_rows.get(basename)
            if not rows:
                skipped += 1
                continue

            img_bytes = zf.read(info)
            for row_id, msgs_json in rows:
                ids.append(row_id)
                messages.append(msgs_json)
                images.append([(basename, img_bytes)])
                matched += 1

            if len(ids) >= BATCH_SIZE:
                table = pa.table({"id": ids, "messages": messages, "images": images}, schema=SCHEMA)
                writer.write_table(table)
                print(f"  Written {matched} rows so far (skipped {skipped} images)...")
                ids, messages, images = [], [], []

        if ids:
            table = pa.table({"id": ids, "messages": messages, "images": images}, schema=SCHEMA)
            writer.write_table(table)

    print(f"Done. Total rows written: {matched}, images without JSONL entry: {skipped}")
    print(f"Output: {OUT_FILE}")


if __name__ == "__main__":
    main()
