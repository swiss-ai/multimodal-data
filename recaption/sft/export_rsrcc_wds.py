#!/usr/bin/env python3
"""Export google_rsrcc candidates to WebDataset tar shards.

Reads all candidates_task*.jsonl from the candidates dir, assembles each
sample into {key}.before.png / {key}.after.png / {key}.json, and writes
one gzip-compressed tar shard per task file.

Each JSON entry contains:
  question   — verbatim from source CSV
  answer     — verbatim from source CSV
  reasoning  — generated CoT (no <think> tags)
  messages   — SFT-ready conversation:
                 user: "<image_0>\\n<image_1>\\n{question}"
                 assistant: "<think>{reasoning}</think>\\n{answer}"

Usage:
    .venv/bin/python scripts/export_rsrcc_wds.py [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sft_recaption.loaders import create_loader
from sft_recaption.wds_export import export_rsrcc_wds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for WDS tar shards (default: artifacts/wds/google_rsrcc/)",
    )
    args = parser.parse_args()

    loader = create_loader("google_rsrcc")
    print(f"Exporting {loader.name} candidates to WebDataset…")
    if args.output_dir:
        print(f"Output dir: {args.output_dir}")

    summary = export_rsrcc_wds(loader, output_dir=args.output_dir, verbose=True)

    print(
        f"\nDone. {summary['shards_written']} shards, "
        f"{summary['total_written']} samples written, "
        f"{summary['total_skipped']} skipped."
    )
    print(f"Output: {summary['output_dir']}")


if __name__ == "__main__":
    main()
