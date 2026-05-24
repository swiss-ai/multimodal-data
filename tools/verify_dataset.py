import os
import sys

import datasets

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DATASET_PATH", "")
if not path:
    print("Usage: python verify_dataset.py <dataset_path>", file=sys.stderr)
    print("  or set DATASET_PATH environment variable", file=sys.stderr)
    sys.exit(1)

dataset_paths = [os.path.join(path, d) for d in os.listdir(path)]

for dataset_path in dataset_paths:
    dataset = datasets.load_from_disk(dataset_path)
    print(f"Dataset at {dataset_path} has {len(dataset)} entries.")
    print(dataset[0])
    print()
