#!/usr/bin/env python3
"""
Export pixmo-ask-model-anything preprocessed data.

Joins:
  - downloaded image shards (url, width, height, jpg where status='success')
  - preprocessed metadata (image_url, messages)

Output schema per shard:
  image_url: string
  messages:  list<struct<role: string, content: string>>
  width:     int32
  height:    int32
  jpg:       binary

Existing shards in OUTPUT_DIR are overwritten.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DOWNLOADED_DIR = Path("/path/to/data/vision-datasets/processed/hf___allenai___pixmo-ask-model-anything___downloaded")
METADATA_PATH = Path("/tmp/metadata/pixmo-ask-model-anything/processed/metadata.parquet")
OUTPUT_DIR = Path("/path/to/data/vision-datasets/raw/sft/hf___allenai___pixmo-ask-model-anything/preprocessed")

MESSAGE_TYPE = pa.struct([("role", pa.string()), ("content", pa.string())])
OUTPUT_SCHEMA = pa.schema(
    [
        ("image_url", pa.string()),
        ("messages", pa.list_(MESSAGE_TYPE)),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("jpg", pa.binary()),
    ]
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("loading metadata...", flush=True)
    meta = pq.read_table(METADATA_PATH).to_pydict()
    url_to_messages = {url: msgs for url, msgs in zip(meta["image_url"], meta["messages"])}
    print(f"  {len(url_to_messages)} metadata rows", flush=True)

    shards = sorted(DOWNLOADED_DIR.glob("*.parquet"))
    total_kept = 0
    total_skipped = 0

    for shard_idx, shard_path in enumerate(shards):
        shard = pq.read_table(shard_path, columns=["url", "status", "width", "height", "jpg"]).to_pydict()

        # One output row per unique url in this shard (deduplicate within shard)
        seen = set()
        out = {k: [] for k in ["image_url", "messages", "width", "height", "jpg"]}

        for url, status, width, height, jpg in zip(
            shard["url"], shard["status"], shard["width"], shard["height"], shard["jpg"]
        ):
            if status != "success" or url not in url_to_messages or url in seen:
                total_skipped += 1
                continue
            seen.add(url)
            out["image_url"].append(url)
            out["messages"].append(url_to_messages[url])
            out["width"].append(width)
            out["height"].append(height)
            out["jpg"].append(jpg)
            total_kept += 1

        out_path = OUTPUT_DIR / f"{shard_idx:04d}.parquet"
        table = pa.table(
            {
                "image_url": pa.array(out["image_url"], type=pa.string()),
                "messages": pa.array(out["messages"], type=pa.list_(MESSAGE_TYPE)),
                "width": pa.array(out["width"], type=pa.int32()),
                "height": pa.array(out["height"], type=pa.int32()),
                "jpg": pa.array(out["jpg"], type=pa.binary()),
            },
            schema=OUTPUT_SCHEMA,
        )
        pq.write_table(table, out_path)
        print(
            f"  shard {shard_idx:04d}: {len(out['image_url'])} rows -> {out_path.name}",
            flush=True,
        )

    print(f"done: kept={total_kept} skipped={total_skipped}", flush=True)


if __name__ == "__main__":
    main()
