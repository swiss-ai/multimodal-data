from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Any

from sft_recaption.loaders.base import BaseLoader
from sft_recaption.schemas import ImagePayload, ManifestRecord, SourceSample

from datasets import Dataset, DatasetDict


class Mint1TArxivProcessedLoader(BaseLoader):
    name = "mint_1t_arxiv_processed"
    dataset_path = "/path/to/data/vision-datasets/processed/hf___mlfoundations___MINT-1T-ArXiv___processed"
    default_split = "train"
    valid_suffixes = {".jpg", ".jpeg", ".png", ".webp"}

    def dataset_root(self) -> Path:
        return Path(os.environ.get("SFT_RECAPTION_MINT_DATASET_PATH", self.dataset_path))

    def iter_shard_paths(self) -> list[Path]:
        root = self.dataset_root()
        shard_paths = sorted(root.glob("shard_*.tar"))
        max_shards = os.environ.get("SFT_RECAPTION_MINT_MAX_SHARDS")
        if max_shards is not None:
            shard_paths = shard_paths[: int(max_shards)]
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.tar files found under {root}")
        return shard_paths

    def load_dataset(self) -> Dataset | DatasetDict:
        raise NotImplementedError("mint_1t_arxiv_processed uses tar shards directly; use prepare-manifests/generate")

    def write_manifests(
        self,
        output_dir: Path,
        task_count: int,
        split: str | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_split = split or self.default_split
        handles: list[tuple[Path, Any]] = []
        row_index = 0
        try:
            for task_id in range(task_count):
                path = output_dir / f"manifest_task{task_id:04d}.jsonl"
                handles.append((path, path.open("w", encoding="utf-8")))
            for shard_path in self.iter_shard_paths():
                with tarfile.open(shard_path, mode="r") as handle:
                    for member in handle:
                        if not member.isfile():
                            continue
                        suffix = Path(member.name).suffix.lower()
                        if suffix not in self.valid_suffixes:
                            continue
                        sample_id = Path(member.name).stem
                        record = ManifestRecord(
                            sample_id=sample_id,
                            split=target_split,
                            row_index=row_index,
                            metadata={
                                "tar_path": str(shard_path),
                                "member_name": member.name,
                                "shard_name": shard_path.name,
                            },
                        )
                        task_id = self.stable_task_id(sample_id, task_count)
                        handles[task_id][1].write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                        row_index += 1
        finally:
            for _, handle in handles:
                handle.close()
        return [path for path, _ in handles]

    def get_source_sample(self, record: ManifestRecord) -> SourceSample:
        if record.metadata is None:
            raise ValueError("MINT manifest record is missing tar shard metadata")
        tar_path = record.metadata["tar_path"]
        member_name = record.metadata["member_name"]
        shard_name = record.metadata["shard_name"]
        with tarfile.open(tar_path, mode="r") as handle:
            file_obj = handle.extractfile(member_name)
            if file_obj is None:
                raise FileNotFoundError(f"Could not extract {member_name} from shard {tar_path}")
            image_bytes = file_obj.read()
        image = ImagePayload(
            media_type=self.media_type_from_path(member_name),
            data=image_bytes,
        )
        metadata = {
            "source_dataset": self.name,
            "source_split": record.split,
            "source_doc_id": record.sample_id,
            "data_source": "mlfoundations/MINT-1T-ArXiv processed",
            "license": None,
            "url": None,
            "source_fields_json": json.dumps(
                {
                    "shard_name": shard_name,
                    "member_name": member_name,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        return SourceSample(
            sample_id=record.sample_id,
            split=record.split,
            row_index=record.row_index,
            images=[image],
            metadata=metadata,
        )

    def build_sample_id(self, split: str, row_index: int, row: dict[str, Any]) -> str:
        del split, row_index
        return str(row["sample_id"])

    def extract_images(self, row: dict[str, Any]) -> list[ImagePayload]:
        return [self.encode_image(row["image"])]
