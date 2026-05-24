#!/usr/bin/env python3
"""Generate a flat per-tar task list for tar-parallel recovery runs."""

import json
import sys
from pathlib import Path

WORK_DIR = Path("/tmp/recaption_blip3")


def find_incomplete_chunks(n_chunks: int) -> list[int]:
    log_dir = WORK_DIR / "logs"
    return [
        i
        for i in range(n_chunks)
        if not (log_dir / f"chunk_{i}.log").exists() or "done," not in (log_dir / f"chunk_{i}.log").read_text()
    ]


def main():
    chunk_map_path = WORK_DIR / "chunk_map_800.json"
    with open(chunk_map_path) as f:
        chunk_map = json.load(f)

    n_chunks = len(chunk_map)
    incomplete = find_incomplete_chunks(n_chunks)
    print(f"Incomplete chunks: {len(incomplete)}", file=sys.stderr)

    tasks = []
    for chunk_id in incomplete:
        tars = chunk_map[str(chunk_id)]
        for i in range(len(tars)):
            tasks.append({"chunk_id": chunk_id, "tar_start": i, "tar_end": i + 1})

    out_path = WORK_DIR / "task_list_recovery.json"
    with open(out_path, "w") as f:
        json.dump(tasks, f)

    n_array = (len(tasks) + 3) // 4
    print(f"Total tar-tasks: {len(tasks)}", file=sys.stderr)
    print(f"Array tasks needed (4 GPUs/node): {n_array}", file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"Submit with: --array=0-{n_array - 1}", file=sys.stderr)


if __name__ == "__main__":
    main()
