"""Loader for HuggingFace .arrow shard files."""

import os
import glob

from datasets import Dataset, Audio


def scan_arrow(input_dir: str):
    files = sorted(glob.glob(os.path.join(input_dir, "*.arrow")))

    def load(path: str) -> Dataset:
        ds = Dataset.from_file(path)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds

    return files, load