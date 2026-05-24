#!/usr/bin/env python3
"""Export google_rsrcc candidates to Parquet shards.

Reads all candidates_task*.jsonl, assembles SFT-ready messages
(<think>{reasoning}</think>\n{answer}), loads before/after images, and
writes one Parquet file per task shard using the same schema as the
arxiv parquet export (images as binary list, messages as struct list).

Usage:
    .venv/bin/python scripts/export_rsrcc_parquet.py [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow as pa
import pyarrow.parquet as pq
from sft_recaption.config import CANDIDATES_DIR
from sft_recaption.loaders import create_loader

SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("source_sample_id", pa.string()),
        ("before", pa.binary()),
        ("after", pa.binary()),
        ("question", pa.string()),
        ("answer", pa.string()),
        ("reasoning", pa.string()),
        (
            "metadata",
            pa.struct(
                [
                    ("source_dataset", pa.string()),
                    ("source_split", pa.string()),
                    ("data_source", pa.string()),
                    ("generator_model", pa.string()),
                    ("prompt_version", pa.string()),
                    ("created_at_utc", pa.string()),
                ]
            ),
        ),
    ]
)


def _process_shard(
    candidate_path: Path,
    output_path: Path,
) -> dict[str, int]:
    rows: list[dict] = []
    skipped = 0

    with candidate_path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            messages = candidate.get("messages")
            metadata = candidate.get("metadata") or {}
            if not isinstance(messages, list) or len(messages) != 2:
                skipped += 1
                continue

            reasoning = messages[1].get("content", "").strip()
            if not reasoning:
                skipped += 1
                continue

            try:
                src = json.loads(metadata.get("source_fields_json", "{}"))
            except json.JSONDecodeError:
                skipped += 1
                continue

            question = src.get("question", "")
            answer = src.get("answer", "")
            before_path = src.get("before_path", "")
            after_path = src.get("after_path", "")

            if not (question and answer and before_path and after_path):
                skipped += 1
                continue

            try:
                before_bytes = Path(before_path).read_bytes()
                after_bytes = Path(after_path).read_bytes()
            except OSError:
                skipped += 1
                continue

            rows.append(
                {
                    "sample_id": candidate.get("sample_id", ""),
                    "source_sample_id": candidate.get("source_sample_id", ""),
                    "before": before_bytes,
                    "after": after_bytes,
                    "question": question,
                    "answer": answer,
                    "reasoning": reasoning,
                    "metadata": {
                        "source_dataset": metadata.get("source_dataset", ""),
                        "source_split": metadata.get("source_split", ""),
                        "data_source": metadata.get("data_source", ""),
                        "generator_model": metadata.get("generator_model", ""),
                        "prompt_version": metadata.get("prompt_version", ""),
                        "created_at_utc": metadata.get("created_at_utc", ""),
                    },
                }
            )

    if rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(table, output_path, compression="zstd")

    return {"written": len(rows), "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: artifacts/parquet/google_rsrcc/)",
    )
    args = parser.parse_args()

    loader = create_loader("google_rsrcc")
    candidates_dir = CANDIDATES_DIR / loader.name
    output_dir = args.output_dir or (CANDIDATES_DIR.parent / "parquet" / loader.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = sorted(candidates_dir.glob("candidates_task*.jsonl"))
    if not candidate_paths:
        print(f"No candidates found under {candidates_dir}. Run generate first.")
        sys.exit(1)

    print(f"Exporting {loader.name} → {output_dir}")
    total_written = total_skipped = 0

    for candidate_path in candidate_paths:
        task_suffix = candidate_path.stem.replace("candidates_", "")
        output_path = output_dir / f"{task_suffix}.parquet"
        stats = _process_shard(candidate_path, output_path)
        total_written += stats["written"]
        total_skipped += stats["skipped"]
        status = "✓" if stats["written"] > 0 else "∅"
        print(f"  {status}  {task_suffix}.parquet: {stats['written']} rows, {stats['skipped']} skipped")

    summary = {
        "total_written": total_written,
        "total_skipped": total_skipped,
        "parquet_files": len(list(output_dir.glob("*.parquet"))),
        "output_dir": str(output_dir),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"\nDone. {total_written:,} rows across {summary['parquet_files']} files, "
        f"{total_skipped} skipped → {output_dir}"
    )


if __name__ == "__main__":
    main()
