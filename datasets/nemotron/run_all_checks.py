#!/usr/bin/env python3
"""Run check_dataset.py for all datasets and collect results into a summary TSV."""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ORIG_BASE = Path(
    "/path/to/data/vision-datasets/raw/sft/nemotron_image_training_v3/hf___nvidia___Nemotron-Image-Training-v3"
)
PYTHON = Path(__file__).parent / ".venv" / "bin" / "python"
CHECK_SCRIPT = Path(__file__).parent / "check_dataset.py"

HEADER = "dataset\tvalid_structure\tvalid_samples\treferences_integrity\tn_samples\tn_mirrored_samples\tn_missing\tpct_coverage"


def get_datasets():
    return sorted(d.name for d in ORIG_BASE.iterdir() if d.is_dir() and d.name != "README.md")


def check_one(dataset, log_dir, python_bin):
    result = subprocess.run(
        [python_bin, str(CHECK_SCRIPT), dataset, "--log-dir", str(log_dir)],
        capture_output=True,
        text=True,
    )
    return dataset, result


def main():
    parser = argparse.ArgumentParser(description="Run checks for all datasets.")
    parser.add_argument("--output", default="summary.tsv", help="Output TSV file path")
    parser.add_argument("--log-dir", default="logs", help="Directory for per-dataset mismatch logs")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    datasets = get_datasets()
    print(
        f"Found {len(datasets)} datasets, checking with {args.workers} workers.",
        file=sys.stderr,
    )

    python_bin = str(PYTHON) if PYTHON.exists() else sys.executable

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_one, d, log_dir, python_bin): d for d in datasets}
        done = 0
        for future in as_completed(futures):
            dataset, result = future.result()
            done += 1
            if result.returncode != 0:
                print(
                    f"[{done}/{len(datasets)}] ERROR {dataset}: {result.stderr.strip()}",
                    file=sys.stderr,
                    flush=True,
                )
                results[dataset] = f"{dataset}\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR\tERROR"
            else:
                print(
                    f"[{done}/{len(datasets)}] done  {dataset}: {result.stdout.strip()}",
                    file=sys.stderr,
                    flush=True,
                )
                results[dataset] = result.stdout.strip()
                if result.stderr:
                    print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write(HEADER + "\n")
        for dataset in datasets:
            f.write(results[dataset] + "\n")

    print(f"\nSummary written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
