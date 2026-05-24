#!/usr/bin/env python3
"""Export JSONL candidate captions to Parquet with embedded images.

Usage:
    .venv/bin/python export_parquet.py [--stage candidates]
"""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HQ50K_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/hf___YangQiee___HQ-50K/downloaded")
CANDIDATES_DIR = Path(__file__).parent / "artifacts" / "candidates"
EXPORT_DIR = Path(__file__).parent / "artifacts" / "parquet"


SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("source_sample_id", pa.string()),
        ("task_type", pa.string()),
        ("images", pa.list_(pa.binary())),
        ("image_media_types", pa.list_(pa.string())),
        (
            "messages",
            pa.list_(
                pa.struct(
                    [
                        ("role", pa.string()),
                        ("content", pa.string()),
                    ]
                )
            ),
        ),
        (
            "metadata",
            pa.struct(
                [
                    ("source_dataset", pa.string()),
                    ("data_source", pa.string()),
                    ("shard", pa.string()),
                    ("member", pa.string()),
                    ("generator_model", pa.string()),
                    ("prompt_version", pa.string()),
                    ("created_at_utc", pa.string()),
                ]
            ),
        ),
    ]
)


def load_image(shard: str, member: str) -> bytes | None:
    shard_path = HQ50K_ROOT / shard
    if not shard_path.exists():
        return None
    try:
        with tarfile.open(shard_path, "r") as tf:
            fobj = tf.extractfile(member)
            return fobj.read() if fobj else None
    except Exception:
        return None


def parse_candidates(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(CANDIDATES_DIR.glob("candidates_task*.jsonl"))
    if not candidate_files:
        print(f"No candidate files found in {CANDIDATES_DIR}")
        return

    total_rows = 0
    source_ids: set[str] = set()

    for cand_file in candidate_files:
        records = parse_candidates(cand_file)
        if not records:
            continue

        rows = []
        for rec in records:
            meta = rec.get("metadata", {})
            shard = meta.get("shard", "")
            member = meta.get("member", "")
            img_bytes = load_image(shard, member)
            if img_bytes is None:
                print(f"  Skipping {rec['sample_id']}: could not load image")
                continue
            rows.append(
                {
                    "sample_id": rec["sample_id"],
                    "source_sample_id": rec["source_sample_id"],
                    "task_type": rec["task_type"],
                    "images": [img_bytes],
                    "image_media_types": ["image/jpeg"],
                    "messages": rec["messages"],
                    "metadata": {
                        "source_dataset": meta.get("source_dataset", ""),
                        "data_source": meta.get("data_source", ""),
                        "shard": shard,
                        "member": member,
                        "generator_model": meta.get("generator_model", ""),
                        "prompt_version": meta.get("prompt_version", ""),
                        "created_at_utc": meta.get("created_at_utc", ""),
                    },
                }
            )
            source_ids.add(rec["sample_id"])

        if not rows:
            continue

        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        out_path = EXPORT_DIR / f"{cand_file.stem}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        total_rows += len(rows)
        print(f"  {cand_file.name} → {out_path.name}  ({len(rows)} rows)")

    summary = {
        "total_rows": total_rows,
        "unique_samples": len(source_ids),
        "parquet_files": len(list(EXPORT_DIR.glob("*.parquet"))),
        "exported_at_utc": datetime.now(UTC).isoformat(),
    }
    (EXPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nExported {total_rows} rows → {EXPORT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
