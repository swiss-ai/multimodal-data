#!/usr/bin/env python3
"""Scan all data2 parquet files and index gpt turns above a character threshold."""

import argparse
import json
import multiprocessing
from pathlib import Path

import pyarrow.parquet as pq

DATA2_DIR = Path("/path/to/data/vision-datasets/raw/sft/hf___InnovatorLab___Innovator-VL-Instruct-46M/data2")


def scan_file(args):
    filepath, threshold = args
    records = []
    try:
        t = pq.read_table(str(filepath), columns=["id", "conversations"])
        d = t.to_pydict()
        for row_idx, (rid, convs) in enumerate(zip(d["id"], d["conversations"])):
            for turn_idx, turn in enumerate(convs):
                if turn["from"] != "gpt" or len(turn["value"]) < threshold:
                    continue
                # last human turn before this gpt turn
                question = ""
                for prev in reversed(convs[:turn_idx]):
                    if prev["from"] == "human":
                        question = prev["value"]
                        break
                records.append(
                    {
                        "file": filepath.name,
                        "row_idx": row_idx,
                        "turn_idx": turn_idx,
                        "char_len": len(turn["value"]),
                        "id": rid,
                        "question": question,
                        "answer": turn["value"],
                    }
                )
    except Exception as e:
        print(f"  error {filepath.name}: {e}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=500)
    parser.add_argument("--output", type=str, default="outputs/long_responses.jsonl")
    parser.add_argument("--workers", type=int, default=256)
    args = parser.parse_args()

    files = sorted(DATA2_DIR.glob("*.parquet"))
    print(
        f"Scanning {len(files)} data2 parquet files with threshold={args.threshold} chars...",
        flush=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    buckets = [500, 1000, 2000, 5000, 10000]
    counts = {b: 0 for b in buckets}

    tasks = [(f, args.threshold) for f in files]

    with open(output_path, "w") as out_f:
        with multiprocessing.Pool(processes=args.workers) as pool:
            for i, records in enumerate(pool.imap_unordered(scan_file, tasks, chunksize=4), 1):
                for rec in records:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
                    for b in buckets:
                        if rec["char_len"] >= b:
                            counts[b] += 1
                if i % 500 == 0:
                    print(
                        f"  {i}/{len(files)} files done, {total} records so far",
                        flush=True,
                    )

    print(f"\nDone. {total} records written to {output_path}")
    print("\nLength distribution:")
    for b in buckets:
        print(f"  >= {b:6d} chars: {counts[b]:8d} ({100 * counts[b] / total:.1f}%)" if total else f"  >= {b}: 0")


if __name__ == "__main__":
    main()
