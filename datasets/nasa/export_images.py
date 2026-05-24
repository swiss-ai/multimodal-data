#!/usr/bin/env python3
"""Export the first 10 images from the NASA images parquet dataset into data/."""

import json
import os
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path(
    os.path.join(
        os.environ.get("DATA_ROOT", "./data"),
        "vision-datasets/raw/cooldown/web___nasa___images/shards/nasa_images_00001.parquet",
    )
)
DST = Path(__file__).parent / "data"
DST.mkdir(exist_ok=True)

N = 10

pf = pq.ParquetFile(SRC)

# Read only the columns we need, batch by batch, stop after N rows
metadata = []
count = 0

for batch in pf.iter_batches(
    batch_size=N,
    columns=["nasa_id", "image_url", "image_bytes", "title", "description"],
):
    for i in range(len(batch)):
        if count >= N:
            break
        nasa_id = batch["nasa_id"][i].as_py()
        image_url = batch["image_url"][i].as_py()
        image_bytes = batch["image_bytes"][i].as_py()
        title = batch["title"][i].as_py()
        description = batch["description"][i].as_py()

        # Derive extension from URL or default to .jpg
        ext = os.path.splitext(image_url.split("?")[0])[-1] or ".jpg"
        safe_id = nasa_id.replace("/", "_")
        filename = f"{count:02d}_{safe_id}{ext}"
        img_path = DST / filename
        img_path.write_bytes(image_bytes)

        metadata.append(
            {
                "index": count,
                "nasa_id": nasa_id,
                "title": title,
                "description": description,
                "image_url": image_url,
                "filename": filename,
            }
        )
        print(f"[{count + 1}/{N}] Saved {filename} ({len(image_bytes):,} bytes)")
        count += 1

    if count >= N:
        break

# Write metadata sidecar
meta_path = DST / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))
print(f"\nDone. {count} images saved to {DST}/")
print(f"Metadata written to {meta_path}")
