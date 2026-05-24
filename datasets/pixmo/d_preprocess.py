#!/usr/bin/env python3
"""
Preprocess pixmo-ask-model-anything into multi-turn messages format.

Multiple independent QA pairs for the same image_url are merged into a
single multi-turn conversation:
  [user: q1, assistant: a1, user: q2, assistant: a2, ...]

Output schema: image_url, messages (list<struct<role, content>>)
"""

from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ORIG_PATH = Path(
    "/path/to/data/vision-datasets/raw/sft/hf___allenai___pixmo-ask-model-anything/data/train-00000-of-00001.parquet"
)
DOWNLOADED_DIR = Path("/path/to/data/vision-datasets/processed/hf___allenai___pixmo-ask-model-anything___downloaded")
OUTPUT_DIR = Path("/tmp/metadata/pixmo-ask-model-anything/processed")
OUTPUT_PATH = OUTPUT_DIR / "metadata.parquet"

MESSAGE_TYPE = pa.struct([("role", pa.string()), ("content", pa.string())])
OUTPUT_SCHEMA = pa.schema(
    [
        ("image_url", pa.string()),
        ("messages", pa.list_(MESSAGE_TYPE)),
    ]
)


def load_downloaded_urls(downloaded_dir):
    urls = set()
    for shard in sorted(downloaded_dir.glob("*.parquet")):
        t = pq.read_table(shard, columns=["url", "status"]).to_pydict()
        for url, status in zip(t["url"], t["status"]):
            if status == "success":
                urls.add(url)
    print(f"loaded {len(urls)} successfully downloaded URLs", flush=True)
    return urls


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_DIR / f"{OUTPUT_PATH.name}.tmp"

    if OUTPUT_PATH.exists():
        print(f"skip {OUTPUT_PATH.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    downloaded_urls = load_downloaded_urls(DOWNLOADED_DIR)

    print("loading original parquet...", flush=True)
    t = pq.read_table(ORIG_PATH, columns=["image_url", "question", "answer"]).to_pydict()

    # Group by image_url, preserving original row order, filter to downloaded
    groups = defaultdict(list)
    dropped = 0
    for url, q, a in zip(t["image_url"], t["question"], t["answer"]):
        if url not in downloaded_urls:
            dropped += 1
            continue
        groups[url].append((q.strip(), a.strip()))

    print(f"unique urls: {len(groups)}  dropped (not downloaded): {dropped}", flush=True)

    out_urls = []
    out_messages = []
    for url, pairs in groups.items():
        msgs = []
        for q, a in pairs:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        out_urls.append(url)
        out_messages.append(msgs)

    table = pa.table(
        {
            "image_url": pa.array(out_urls, type=pa.string()),
            "messages": pa.array(out_messages, type=pa.list_(MESSAGE_TYPE)),
        },
        schema=OUTPUT_SCHEMA,
    )
    pq.write_table(table, temp_path)
    temp_path.rename(OUTPUT_PATH)
    print(f"done: {len(out_urls)} rows -> {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
