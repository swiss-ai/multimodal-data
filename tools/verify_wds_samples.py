import glob
import os
import random
import shutil

import webdataset as wds

TARS = 4
SAMPLES = 5

random.seed(42)

with open("../paths/paths.txt", "r") as f:
    paths = [line.strip() for line in f if line.strip()]

print(f"Found {len(paths)} paths")
sampling_tars = []
for path in paths:
    if not os.path.isdir(path):
        print(f"Path does not exist: {path}")
        exit(1)
    tar_files = glob.glob(os.path.join(path, "**", "*.tar"), recursive=True)
    if not tar_files:
        print(f"No .tar files found in: {path}")
        continue
    random_tars = random.sample(tar_files, min(TARS, len(tar_files)))
    full_paths = [(path, os.path.join(path, tar)) for tar in random_tars]
    sampling_tars.extend(full_paths)

print(f"Sampling from {len(sampling_tars)} .tar files")
shutil.rmtree("sampled", ignore_errors=True)
for parent_path, tar in sampling_tars:
    print(f"Sampling from: {tar}")
    dataset = wds.WebDataset(tar, shardshuffle=False)  # type: ignore
    path = os.path.join(
        "sampled",
        os.path.basename(parent_path.rstrip("/")),
        os.path.basename(tar).replace(".tar", ""),
    )
    os.makedirs(path, exist_ok=True)
    for i, sample in enumerate(dataset):
        if i >= SAMPLES:
            break
        sample_path = os.path.join(path, f"{i:04d}-{sample['__key__']}")
        for key, value in sample.items():
            if key.startswith("__"):
                continue
            if key == "txt":
                open(sample_path + ".txt", "wb").write(value)
            else:
                open(sample_path + f".{key}", "wb").write(value)
