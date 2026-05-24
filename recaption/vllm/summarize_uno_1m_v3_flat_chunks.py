#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

OUTPUT_ROOT = Path("/path/to/data/vision-datasets/UNO-1M___paired_recap_v3")


def main() -> None:
    chunk_paths = sorted(OUTPUT_ROOT.glob("chunk_summary_*.json"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk summaries found under {OUTPUT_ROOT}")

    chunks = [json.loads(path.read_text(encoding="utf-8")) for path in chunk_paths]
    summary = {
        "chunk_count": len(chunks),
        "written": sum(chunk["written"] for chunk in chunks),
        "captions_loaded": sum(chunk["captions_loaded"] for chunk in chunks),
        "unused_captions": sum(chunk["unused_captions"] for chunk in chunks),
        "shard_count": len(list(OUTPUT_ROOT.glob("part-*.tar"))),
        "chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "written": chunk["written"],
                "assigned_splits": chunk["assigned_splits"],
            }
            for chunk in chunks
        ],
    }
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
