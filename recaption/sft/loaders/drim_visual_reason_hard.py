from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sft_recaption.config import WORKDIR
from sft_recaption.loaders.base import BaseLoader
from sft_recaption.schemas import ImagePayload

from datasets import Dataset, DatasetDict, Image, load_dataset


class DrimVisualReasonHardLoader(BaseLoader):
    name = "drim_visual_reason_hard"
    dataset_path = "/path/to/data/vision-datasets/hf___xiuhuywh___DRIM-VisualReasonHard"
    default_split = "train"

    def candidate_dataset_paths(self) -> list[Path]:
        override = os.environ.get("SFT_RECAPTION_DRIM_DATASET_PATH")
        paths: list[Path] = []
        if override:
            paths.append(Path(override))
        paths.append(Path(self.dataset_path))
        paths.append(WORKDIR / "artifacts" / "hf" / self.name)
        return paths

    def load_dataset(self) -> Dataset | DatasetDict:
        last_error: Exception | None = None
        for dataset_path in self.candidate_dataset_paths():
            try:
                self.dataset_path = str(dataset_path)
                return super().load_dataset()
            except FileNotFoundError as exc:
                last_error = exc
                data_dir = dataset_path / "data"
                if not data_dir.exists():
                    continue
                dataset = load_dataset("imagefolder", data_dir=str(data_dir))
                if "train" not in dataset:
                    raise FileNotFoundError(f"No train split found under DRIM data directory {data_dir}")
                return dataset
        raise FileNotFoundError("DRIM dataset could not be resolved from any supported path") from last_error

    def build_sample_id(self, split: str, row_index: int, row: dict[str, Any]) -> str:
        sample_id = row.get("sample_id")
        if sample_id is not None:
            return str(sample_id)
        doc_id = row.get("doc_id")
        if doc_id is not None:
            return f"{split}:{doc_id}"
        image_value = row.get("image")
        if isinstance(image_value, dict) and image_value.get("path"):
            return f"{split}:{Path(image_value['path']).stem}"
        return f"{split}:{row_index}"

    def get_manifest_dataset(self, split: str | None = None) -> Dataset:
        dataset = self.get_split_dataset(split)
        if "image" in dataset.column_names:
            return dataset.cast_column("image", Image(decode=False))
        return dataset

    def extract_images(self, row: dict[str, Any]) -> list[ImagePayload]:
        images = row.get("images") or []
        if isinstance(images, list) and images:
            return [self.encode_image(image) for image in images]
        image_value = row.get("image")
        if image_value is not None:
            return [self.encode_image(image_value)]
        raise ValueError("DRIM sample does not contain a usable image payload")

    def extract_source_doc_id(self, row: dict[str, Any]) -> str | None:
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("source_doc_id"):
            return str(metadata["source_doc_id"])
        value = row.get("doc_id")
        if value is None:
            image_value = row.get("image")
            if isinstance(image_value, dict) and image_value.get("path"):
                return Path(image_value["path"]).stem
        return str(value) if value is not None else None

    def extract_data_source(self, row: dict[str, Any]) -> str | None:
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("data_source"):
            return str(metadata["data_source"])
        value = row.get("data_source")
        if value is not None:
            return str(value)
        return "hf___xiuhuywh___DRIM-VisualReasonHard"
