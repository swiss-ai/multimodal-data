#!/usr/bin/env python3
"""Map dataset tar files to N chunks for parallel recaptioning."""

import argparse
import json
from pathlib import Path

DATASET_DIR = Path("/path/to/data/vision-datasets/raw/stage2/hf___Salesforce___blip3-grounding-50m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-chunks", type=int, default=400)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-tars", type=int, default=None, help="Limit to first N tars (for testing)")
    args = parser.parse_args()

    tar_paths = sorted(str(p) for p in DATASET_DIR.glob("**/chunk_*/*.tar"))

    if args.n_tars:
        tar_paths = tar_paths[: args.n_tars]

    n = args.n_chunks
    chunks: dict[int, list[str]] = {i: [] for i in range(n)}
    for i, path in enumerate(tar_paths):
        chunks[i % n].append(path)

    with open(args.output, "w") as f:
        json.dump({str(k): v for k, v in sorted(chunks.items())}, f, indent=2)

    print(f"Wrote {args.output}: {len(tar_paths)} tars → {n} chunks")
    for i in range(min(5, n)):
        print(f"  chunk {i}: {len(chunks[i])} tars")


if __name__ == "__main__":
    main()
