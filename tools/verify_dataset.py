import os

import datasets

path = "/path/to/data/medical/apertus_image_only_v2"
dataset_paths = [os.path.join(path, d) for d in os.listdir(path)]

for dataset_path in dataset_paths:
    dataset = datasets.load_from_disk(dataset_path)
    print(f"Dataset at {dataset_path} has {len(dataset)} entries.")
    print(dataset[0])
    print()
