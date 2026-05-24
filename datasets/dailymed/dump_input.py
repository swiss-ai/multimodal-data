"""
Pre-dump parquet_md shards to a single JSONL file that recaption.py can read
without pyarrow (images encoded as base64 strings).

Run with the dailymed .venv before submitting the recaption job:
    .venv/bin/python dump_input.py
"""

import base64
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

SRC_DIR = os.environ.get(
    "SRC_DIR",
    "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_md",
)
OUT_JSONL = os.environ.get(
    "OUT_JSONL",
    "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_caption/_input.jsonl",
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".svg", ".webp"}


def media_type(name: str) -> str:
    ext = Path(name).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def main() -> None:
    Path(OUT_JSONL).parent.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(SRC_DIR).glob("part-*.parquet"))
    assert shards, f"No shards in {SRC_DIR}"
    print(f"[dump] {len(shards)} shards → {OUT_JSONL}")

    n = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as fh:
        for shard in tqdm(shards, unit="shard"):
            t = pq.read_table(str(shard), columns=["id", "markdown", "images"])
            for row in t.to_pylist():
                images_out = []
                for img in row.get("images") or []:
                    raw: bytes = img["bytes"] if isinstance(img["bytes"], bytes) else bytes(img["bytes"])
                    images_out.append(
                        {
                            "name": img["name"],
                            "media_type": media_type(img["name"]),
                            "b64": base64.b64encode(raw).decode("ascii"),
                        }
                    )
                fh.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "markdown": row.get("markdown") or "",
                            "images": images_out,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1

    print(f"[dump] wrote {n} rows to {OUT_JSONL}")


if __name__ == "__main__":
    main()
