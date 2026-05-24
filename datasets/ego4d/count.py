"""
This script counts the number of images in each category (kept, duplicate, blurry)
across all subdirectories in the specified path. It reads the metadata.json
file in each subdirectory to extract the summary statistics and aggregates them
to compute the overall percentages for each category.
"""

import json
import os

stats = {
    "total": 0,
    "kept": 0,
    "duplicate": 0,
    "blurry": 0,
}


path = "/tmp/ego/sample_A"
for subdir in os.listdir(path):
    subdir_path = os.path.join(path, subdir)
    metadata_path = os.path.join(subdir_path, "metadata.json")
    assert os.path.exists(metadata_path), f"Metadata not found in {subdir_path}"

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    s = metadata["__summary"]
    for key in s.keys():
        stats[key] += s[key]

print("Overall statistics:")
total = stats["total"]
for key, value in stats.items():
    print(f"{key}: {value / total:.2%}")
