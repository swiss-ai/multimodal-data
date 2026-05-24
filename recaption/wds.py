"""WebDataset-backed loader: streams images from .tar shards."""

import base64
import glob
import os
from typing import Iterator

import numpy as np
import webdataset as wds

from .base import Loader


class WdsLoader(Loader):
    """
    Streams b64-encoded images from a directory of WebDataset .tar files.

    The tar files are sorted and split evenly across task_count tasks;
    this instance handles the slice for task_id.

    Args:
        task_id:    SLURM_ARRAY_TASK_ID (0-based)
        task_count: SLURM_ARRAY_TASK_COUNT
        wds_dir:    Directory containing *.tar shards
        image_keys: Keys to extract from each sample, in order
    """

    def __init__(
        self,
        task_id: int,
        task_count: int,
        wds_dir: str,
        image_keys: tuple[str, ...],
    ) -> None:
        super().__init__(task_id, task_count)
        self.image_keys = image_keys

        tar_files = sorted(glob.glob(os.path.join(wds_dir, "*.tar")))
        if not tar_files:
            raise FileNotFoundError(f"No .tar files found in {wds_dir}")

        self.tar_paths: list[str] = list(np.array_split(tar_files, task_count)[task_id])

    def __iter__(self) -> Iterator[tuple[str, list[str]]]:
        """Yield (sample_id, [b64_img, ...]) for every sample across all shards."""
        print(f"Task {self.task_id} processing {len(self.tar_paths)} shards")
        print(f"From {self.tar_paths[0]} to {self.tar_paths[-1]}")
        for sample in wds.WebDataset(self.tar_paths, shardshuffle=False):
            key = sample["__key__"]
            b64_images = [self._to_b64(sample[k], k) for k in self.image_keys]
            yield key, b64_images

    @staticmethod
    def _to_b64(img_bytes: bytes, key: str = "") -> str:
        if img_bytes[:2] != b"\xff\xd8":
            raise ValueError(f"Expected JPEG for {key!r}, got magic {img_bytes[:4].hex()}")
        return base64.b64encode(img_bytes).decode()
