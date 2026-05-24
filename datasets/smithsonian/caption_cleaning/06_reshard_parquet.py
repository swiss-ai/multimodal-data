#!/usr/bin/env python3
"""
06_reshard_parquet.py

Read all WDS tars from smithsonian_cleaned4/ and write flat parquet shards
to smithsonian_cleaned5/, matching the schema of hq50k_v7_flat:

  sample_id: string
  image_bytes: binary
  caption: string

Output files are named 00000.parquet, 00001.parquet, etc.
"""

import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IN_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian/smithsonian_cleaned4")
OUT_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian/smithsonian_cleaned5")
ROWS_PER_SHARD = 500  # ~94 shards for ~47K samples

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("image_bytes", pa.binary()),
        ("caption", pa.string()),
    ]
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def iter_samples(in_root: Path):
    """Yield (sample_id, jpg_bytes, caption) from all tars in in_root."""
    tars = sorted(in_root.glob("*.tar"))
    print(f"Found {len(tars)} tars in {in_root}")
    for tar_path in tars:
        buffers: dict[str, dict] = {}
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if not member.isfile() or "." not in member.name:
                    continue
                stem, ext = member.name.rsplit(".", 1)
                if ext not in ("jpg", "txt"):
                    continue
                f = tf.extractfile(member)
                if f is None:
                    continue
                if stem not in buffers:
                    buffers[stem] = {}
                if ext == "jpg":
                    buffers[stem]["jpg"] = f.read()
                elif ext == "txt":
                    buffers[stem]["txt"] = f.read().decode("utf-8", errors="replace").strip()

        for key, parts in buffers.items():
            if "jpg" in parts and "txt" in parts:
                yield key, parts["jpg"], parts["txt"]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    shard_idx = 0
    batch_ids, batch_imgs, batch_caps = [], [], []
    total = 0

    def flush(final: bool = False) -> None:
        nonlocal shard_idx
        if not batch_ids:
            return
        table = pa.table(
            {"sample_id": batch_ids, "image_bytes": batch_imgs, "caption": batch_caps},
            schema=SCHEMA,
        )
        out_path = OUT_ROOT / f"{shard_idx:05d}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        print(f"  wrote {out_path}  ({len(batch_ids)} rows)")
        shard_idx += 1
        batch_ids.clear()
        batch_imgs.clear()
        batch_caps.clear()

    for sample_id, jpg_bytes, caption in iter_samples(IN_ROOT):
        batch_ids.append(sample_id)
        batch_imgs.append(jpg_bytes)
        batch_caps.append(caption)
        total += 1
        if len(batch_ids) >= ROWS_PER_SHARD:
            flush()

    flush(final=True)

    print(f"\nDone. {total} samples → {shard_idx} parquet shards in {OUT_ROOT}")


if __name__ == "__main__":
    main()
