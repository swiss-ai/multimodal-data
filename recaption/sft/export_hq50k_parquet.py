#!/usr/bin/env python3
"""Export HQ-50K captions to parquet.

Strategy: load all captions into memory keyed by member_name, then walk
each of the 6 tar shards once extracting images. Write one parquet per
task file at the end.

Usage:
    .venv/bin/python scripts/export_hq50k_parquet.py \
        [--output-dir artifacts/parquet/hq50k] \
        [--workers 32]
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

CANDIDATES_DIR = _ROOT / "artifacts" / "candidates" / "hq50k"
HQ50K_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/hf___YangQiee___HQ-50K/downloaded")


def build_parquet_row(candidate: dict, image_bytes: bytes) -> dict:
    meta = candidate.get("metadata", {})
    return {
        "sample_id": candidate["sample_id"],
        "source_sample_id": candidate["source_sample_id"],
        "task_type": candidate["task_type"],
        "image": image_bytes,
        "image_media_type": "image/jpeg",
        "messages": candidate["messages"],
        "quality_score": 0.0,
        "metadata": {
            "source_dataset": meta.get("source_dataset") or "",
            "source_split": meta.get("source_split") or "",
            "source_doc_id": meta.get("source_doc_id") or "",
            "data_source": meta.get("data_source") or "",
            "license": meta.get("license") or "",
            "url": meta.get("url") or "",
            "source_fields_json": meta.get("source_fields_json") or "{}",
            "generator_model": meta.get("generator_model") or "",
            "prompt_version": meta.get("prompt_version") or "",
            "judge_model": meta.get("judge_model") or "",
            "judge_prompt_version": meta.get("judge_prompt_version") or "",
            "created_at_utc": meta.get("created_at_utc") or "",
        },
    }


def write_parquet(rows: list[dict], out_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("source_sample_id", pa.string()),
            ("task_type", pa.string()),
            ("image", pa.binary()),
            ("image_media_type", pa.string()),
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
            ("quality_score", pa.float32()),
            (
                "metadata",
                pa.struct(
                    [
                        ("source_dataset", pa.string()),
                        ("source_split", pa.string()),
                        ("source_doc_id", pa.string()),
                        ("data_source", pa.string()),
                        ("license", pa.string()),
                        ("url", pa.string()),
                        ("source_fields_json", pa.string()),
                        ("generator_model", pa.string()),
                        ("prompt_version", pa.string()),
                        ("judge_model", pa.string()),
                        ("judge_prompt_version", pa.string()),
                        ("created_at_utc", pa.string()),
                    ]
                ),
            ),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, out_path, compression="zstd")


def scan_shard(args: tuple) -> dict[str, list[dict]]:
    """Open one tar shard, extract all images that have a matching candidate."""
    shard_path, member_to_candidate = args
    rows_by_stem: dict[str, list[dict]] = defaultdict(list)
    with tarfile.open(shard_path, mode="r") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".jpg"):
                continue
            entry = member_to_candidate.get(member.name)
            if entry is None:
                continue
            stem, candidate = entry
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            rows_by_stem[stem].append(build_parquet_row(candidate, fobj.read()))
    shard_name = Path(shard_path).name
    total = sum(len(v) for v in rows_by_stem.values())
    print(f"  {shard_name}: {total} images extracted", flush=True)
    return dict(rows_by_stem)


def write_one(args: tuple) -> tuple[str, int]:
    stem, rows, output_dir = args
    out_path = Path(output_dir) / f"{stem}.parquet"
    write_parquet(rows, out_path)
    return stem, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/parquet/hq50k")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load all candidates into memory
    # Index: member_name -> (task_stem, candidate_dict)
    print("Loading all candidates into memory...")
    member_to_candidate: dict[str, tuple[str, dict]] = {}
    task_files = sorted(CANDIDATES_DIR.glob("candidates_task*.jsonl"))
    for tf in task_files:
        for line in tf.read_text().splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            fields = json.loads(c.get("metadata", {}).get("source_fields_json", "{}"))
            member_name = fields.get("member_name")
            if member_name:
                member_to_candidate[member_name] = (tf.stem, c)

    print(f"  {len(member_to_candidate)} candidates across {len(task_files)} task files")

    # Step 2: one worker per shard, all 6 run in parallel
    shards = sorted(HQ50K_ROOT.glob("*.tar"))
    print(f"Scanning {len(shards)} shards in parallel...")

    import multiprocessing as mp

    shard_work = [(str(shard), member_to_candidate) for shard in shards]
    rows_by_stem: dict[str, list[dict]] = defaultdict(list)
    total_found = 0

    with mp.Pool(processes=len(shards)) as pool:
        for shard_rows in pool.imap_unordered(scan_shard, shard_work):
            for stem, rows in shard_rows.items():
                rows_by_stem[stem].extend(rows)
                total_found += len(rows)

    print(f"Extracted {total_found} images. Writing {len(rows_by_stem)} parquet files...")

    import multiprocessing as mp

    work = [(stem, rows, str(output_dir)) for stem, rows in rows_by_stem.items()]
    written = 0
    with mp.Pool(processes=args.workers) as pool:
        for stem, count in pool.imap_unordered(write_one, work):
            written += count
            print(f"  wrote {stem}.parquet ({count} rows)  total={written}", flush=True)

    parquet_files = list(output_dir.glob("*.parquet"))
    print(f"\nDone: {written} rows in {len(parquet_files)} parquet files → {output_dir}")


if __name__ == "__main__":
    main()
