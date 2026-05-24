#!/usr/bin/env python3
"""Split a JSONL file into per-worker chunk files for parallel processing."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input.jsonl")
    parser.add_argument("--n-chunks", type=int, default=800)
    parser.add_argument("--output-dir", type=str, default="chunks")
    args = parser.parse_args()

    print(f"Loading {args.input}...", flush=True)
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records", flush=True)
    random.shuffle(records)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_chunks
    chunk_size = (len(records) + n - 1) // n

    for i in range(n):
        chunk = records[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            break
        chunk_path = output_dir / f"chunk_{i:04d}.jsonl"
        with open(chunk_path, "w") as f:
            for rec in chunk:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    actual_chunks = min(n, (len(records) + chunk_size - 1) // chunk_size)
    print(f"Wrote {actual_chunks} chunks of ~{chunk_size} records each to {output_dir}/")
    print(f"For Slurm: --array=0-{actual_chunks - 1}")


if __name__ == "__main__":
    main()
