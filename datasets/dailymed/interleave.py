"""
Build interleaved parquet shards from parquet_md + task_*.jsonl captions.

For each document, splits the markdown on [[IMAGE: <mid> | <filename>]] anchors
into alternating text/image segments, attaches generated captions, and keeps
the raw image bytes — producing the same interleaved format as owid___charts.

Output schema per row (one row = one document):
    id             str    - doc id
    n_segments     int32  - total segment count
    n_images       int32  - image segment count
    n_text_chars   int32  - total chars across text segments
    segments_json  str    - JSON list of {type, value} or {type, image_index, filename, caption}
    images_bytes   list   - raw image bytes, indexed by image_index in segments_json

Run:
    .venv/bin/python interleave.py
"""

import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

CAPTION_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_caption"
SRC_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_md"
DST_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/interleaved"

IMAGE_RE = re.compile(r"\[\[IMAGE: [^\]|]+ \| ([^\]]+)\]\]")

SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("n_segments", pa.int32()),
        pa.field("n_images", pa.int32()),
        pa.field("n_text_chars", pa.int32()),
        pa.field("segments_json", pa.large_string()),
        pa.field("images_bytes", pa.list_(pa.large_binary())),
    ]
)


def load_captions(caption_dir: str) -> dict:
    caps = {}
    files = sorted(Path(caption_dir).glob("task_*.jsonl"))
    print(f"[interleave] loading captions from {len(files)} files...")
    for jf in tqdm(files, unit="file"):
        with open(jf, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                caps[(r["doc_id"], r["image_name"])] = r["caption"]
    print(f"[interleave] {len(caps)} captions loaded")
    return caps


def parse_segments(markdown: str, doc_id: str, images_by_name: dict, captions: dict):
    # re.split with a capturing group gives [text, filename, text, filename, ..., text]
    parts = IMAGE_RE.split(markdown)

    segments = []
    images_bytes_out = []
    img_idx = 0

    i = 0
    while i < len(parts):
        text = parts[i].strip()
        if len(text) >= 2000:
            segments.append({"type": "text", "value": text})
        i += 1
        if i < len(parts):
            filename = parts[i].strip()
            alt = captions.get((doc_id, filename), "")
            raw = images_by_name.get(filename, b"")
            if not isinstance(raw, bytes):
                raw = bytes(raw)
            segments.append(
                {
                    "type": "image",
                    "image_index": img_idx,
                    "filename": filename,
                    "alt": alt,
                }
            )
            images_bytes_out.append(raw)
            img_idx += 1
            i += 1

    return segments, images_bytes_out


def process_shard(shard_path: Path, out_path: Path, captions: dict) -> None:
    t = pq.read_table(str(shard_path), columns=["id", "markdown", "images"])
    rows = t.to_pylist()

    out_rows = []
    for row in rows:
        doc_id = row["id"]
        markdown = row.get("markdown") or ""
        images_by_name = {img["name"]: img["bytes"] for img in (row.get("images") or [])}

        segments, img_bytes = parse_segments(markdown, doc_id, images_by_name, captions)

        n_images = sum(1 for s in segments if s["type"] == "image")
        n_text_chars = sum(len(s["value"]) for s in segments if s["type"] == "text")

        out_rows.append(
            {
                "id": doc_id,
                "n_segments": len(segments),
                "n_images": n_images,
                "n_text_chars": n_text_chars,
                "segments_json": json.dumps(segments, ensure_ascii=False),
                "images_bytes": img_bytes,
            }
        )

    table = pa.Table.from_pylist(out_rows, schema=SCHEMA)
    pq.write_table(table, str(out_path), compression="zstd", compression_level=3)
    size_mb = out_path.stat().st_size / 1e6
    n_imgs = sum(r["n_images"] for r in out_rows)
    print(f"[interleave] {out_path.name}: {len(out_rows)} docs, {n_imgs} images, {size_mb:.0f} MB")


def main():
    Path(DST_DIR).mkdir(parents=True, exist_ok=True)
    captions = load_captions(CAPTION_DIR)

    shards = sorted(Path(SRC_DIR).glob("part-*.parquet"))
    print(f"[interleave] {len(shards)} shards → {DST_DIR}")

    for shard in tqdm(shards, unit="shard"):
        out_path = Path(DST_DIR) / shard.name
        if out_path.exists():
            print(f"[interleave] {shard.name} already exists, skipping")
            continue
        process_shard(shard, out_path, captions)

    print("[interleave] done")


if __name__ == "__main__":
    main()
