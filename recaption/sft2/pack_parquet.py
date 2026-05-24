#!/usr/bin/env python3
"""
Pack {key}.jpg + {key}.txt + {key}.json triplets into parquet.

Schema:
  sample_id   : string
  image_bytes : binary
  caption     : string
  model       : string
  persona     : string
  style       : string
  metadata    : string  (JSON: caption_chars, image_width, image_height, image_bytes)

One parquet file per shard, split into multiple parts if size > MAX_BYTES.
Output is flat: shard_{NNN}-part_{MMM}.parquet
"""

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_ROOT = Path("/tmp/toolbox/story_caption/outputs/full_run")
OUTPUT_ROOT = Path("/tmp/data/vision-datasets/processed/swisstopo___swissmap___cooldown")

MAX_BYTES = 2 * 1024 * 1024 * 1024
ROW_GROUP_SIZE = 256

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("image_bytes", pa.binary()),
        ("caption", pa.string()),
        ("model", pa.string()),
        ("persona", pa.string()),
        ("style", pa.string()),
        ("metadata", pa.string()),
    ]
)

META_KEEP = ("caption_chars", "image_width", "image_height", "image_bytes")


def iter_samples(shard_dir: Path):
    for jpg_path in sorted(shard_dir.glob("*.jpg")):
        key = jpg_path.stem
        txt_path = shard_dir / f"{key}.txt"
        json_path = shard_dir / f"{key}.json"
        if not txt_path.exists() or not json_path.exists():
            continue
        try:
            full_meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sub_meta = {k: full_meta.get(k) for k in META_KEEP}
        yield {
            "sample_id": key,
            "image_bytes": jpg_path.read_bytes(),
            "caption": txt_path.read_text(encoding="utf-8"),
            "model": full_meta.get("model", ""),
            "persona": full_meta.get("persona", ""),
            "style": full_meta.get("style", ""),
            "metadata": json.dumps(sub_meta),
        }


def row_size(row: dict) -> int:
    return (
        len(row["sample_id"])
        + len(row["image_bytes"])
        + len(row["caption"])
        + len(row["model"])
        + len(row["persona"])
        + len(row["style"])
        + len(row["metadata"])
    )


def pack_shard(shard_idx: int):
    shard_dir = INPUT_ROOT / f"shard_{shard_idx:03d}"
    if not shard_dir.is_dir():
        print(f"  [skip] {shard_dir} not found")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    part_idx = 0
    writer: pq.ParquetWriter | None = None
    out_path: Path | None = None
    bytes_written = 0
    rows_buffered: list[dict] = []
    bytes_buffered = 0
    total_rows = 0

    def open_writer() -> tuple[pq.ParquetWriter, Path]:
        nonlocal part_idx
        path = OUTPUT_ROOT / f"shard_{shard_idx:03d}-part_{part_idx:03d}.parquet"
        w = pq.ParquetWriter(path, SCHEMA, compression="zstd")
        print(f"  [open] {path.name}")
        return w, path

    def flush(w: pq.ParquetWriter):
        nonlocal rows_buffered, bytes_buffered, bytes_written
        if not rows_buffered:
            return
        cols = {name: [] for name in SCHEMA.names}
        for r in rows_buffered:
            for name in SCHEMA.names:
                cols[name].append(r[name])
        table = pa.table(cols, schema=SCHEMA)
        w.write_table(table, row_group_size=ROW_GROUP_SIZE)
        bytes_written += bytes_buffered
        rows_buffered = []
        bytes_buffered = 0

    for row in iter_samples(shard_dir):
        rsz = row_size(row)

        if writer is None:
            writer, out_path = open_writer()
            bytes_written = 0

        if bytes_written + bytes_buffered + rsz > MAX_BYTES and (bytes_written + bytes_buffered > 0):
            flush(writer)
            writer.close()
            assert out_path is not None
            print(f"  [close] {out_path.name} ({bytes_written / 1e9:.2f} GB)")
            part_idx += 1
            writer, out_path = open_writer()
            bytes_written = 0

        rows_buffered.append(row)
        bytes_buffered += rsz
        total_rows += 1

        if bytes_buffered > 256 * 1024 * 1024:
            flush(writer)

    if writer is not None:
        flush(writer)
        writer.close()
        assert out_path is not None
        print(f"  [close] {out_path.name} ({bytes_written / 1e9:.2f} GB)")

    print(f"  shard_{shard_idx:03d}: {total_rows} rows -> {part_idx + 1} part(s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["shard", "slurm"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    if args.mode == "shard":
        pack_shard(args.shard)
        return

    worker_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    num_workers = args.num_workers
    total_shards = 128
    my_shards = [s for s in range(total_shards) if s % num_workers == worker_id]
    print(f"Worker {worker_id}/{num_workers}: shards {my_shards}")
    for s in my_shards:
        pack_shard(s)


if __name__ == "__main__":
    main()
