from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from hashlib import blake2b
from pathlib import Path
from typing import Any

from PIL import Image
from sft_recaption.schemas import ImagePayload, ManifestRecord, SourceSample

from datasets import Dataset, DatasetDict, load_from_disk


class BaseLoader(ABC):
    name: str
    dataset_path: str
    default_split: str = "train"
    ignored_source_fields: set[str] = {"problem", "solution"}

    def load_dataset(self) -> Dataset | DatasetDict:
        return load_from_disk(self.dataset_path)

    def get_split_dataset(self, split: str | None = None) -> Dataset:
        dataset = self.load_dataset()
        target_split = split or self.default_split
        if isinstance(dataset, DatasetDict):
            try:
                return dataset[target_split]
            except KeyError as exc:
                raise KeyError(f"Split {target_split!r} not found for loader {self.name!r}") from exc
        if target_split != self.default_split:
            raise KeyError(f"Loader {self.name!r} exposes a single dataset, not split {target_split!r}")
        return dataset

    def get_manifest_dataset(self, split: str | None = None) -> Dataset:
        return self.get_split_dataset(split)

    def stable_task_id(self, sample_id: str, task_count: int) -> int:
        digest = blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big") % task_count

    def iter_manifest_records(self, split: str | None = None) -> list[ManifestRecord]:
        dataset = self.get_manifest_dataset(split)
        target_split = split or self.default_split
        return [
            ManifestRecord(
                sample_id=self.build_sample_id(target_split, row_index, row),
                split=target_split,
                row_index=row_index,
            )
            for row_index, row in enumerate(dataset)
        ]

    def write_manifests(
        self,
        output_dir: Path,
        task_count: int,
        split: str | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = output_dir / "_source_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = self.get_manifest_dataset(split)
        target_split = split or self.default_split
        handles: list[tuple[Path, Any]] = []
        try:
            for task_id in range(task_count):
                path = output_dir / f"manifest_task{task_id:04d}.jsonl"
                handles.append((path, path.open("w", encoding="utf-8")))
            for row_index, row in enumerate(dataset):
                sample_id = self.build_sample_id(target_split, row_index, row)
                images = self.extract_images(row)
                cached_image_paths = self.cache_images(cache_dir, sample_id, images)
                record = ManifestRecord(
                    sample_id=sample_id,
                    split=target_split,
                    row_index=row_index,
                    cached_image_paths=[str(path.relative_to(output_dir)) for path in cached_image_paths],
                    metadata=self.build_metadata(target_split, row),
                )
                task_id = self.stable_task_id(record.sample_id, task_count)
                handle = handles[task_id][1]
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        finally:
            for _, handle in handles:
                handle.close()
        return [path for path, _ in handles]

    def load_manifest(self, manifest_path: Path) -> list[ManifestRecord]:
        records: list[ManifestRecord] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                cached_image_paths = payload.get("cached_image_paths")
                if cached_image_paths is not None:
                    cached_image_paths = [str((manifest_path.parent / path).resolve()) for path in cached_image_paths]
                records.append(
                    ManifestRecord(
                        sample_id=payload["sample_id"],
                        split=payload["split"],
                        row_index=int(payload["row_index"]),
                        cached_image_paths=cached_image_paths,
                        metadata=payload.get("metadata"),
                    )
                )
        return records

    def get_source_sample(self, record: ManifestRecord) -> SourceSample:
        if record.cached_image_paths is not None and record.metadata is not None:
            return SourceSample(
                sample_id=record.sample_id,
                split=record.split,
                row_index=record.row_index,
                images=[self.encode_image(path) for path in record.cached_image_paths],
                metadata=record.metadata,
            )
        dataset = self.get_split_dataset(record.split)
        row = dataset[record.row_index]
        return SourceSample(
            sample_id=record.sample_id,
            split=record.split,
            row_index=record.row_index,
            images=self.extract_images(row),
            metadata=self.build_metadata(record.split, row),
        )

    def build_metadata(self, split: str, row: dict[str, Any]) -> dict[str, Any]:
        preserved = {
            key: value
            for key, value in row.items()
            if key not in self.ignored_source_fields and key not in {"images", "image"}
        }
        return {
            "source_dataset": self.name,
            "source_split": split,
            "source_doc_id": self.extract_source_doc_id(row),
            "data_source": self.extract_data_source(row),
            "license": row.get("license"),
            "url": row.get("url"),
            "source_fields_json": json.dumps(self.json_safe(preserved), ensure_ascii=False, sort_keys=True),
        }

    def json_safe(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self.json_safe(inner) for key, inner in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def encode_image(self, value: Any) -> ImagePayload:
        if isinstance(value, (str, Path)):
            path = Path(value)
            return ImagePayload(
                media_type=self.media_type_from_path(str(path)),
                data=path.read_bytes(),
            )
        if isinstance(value, dict):
            raw_bytes = value.get("bytes")
            if raw_bytes is not None:
                media_type = self.media_type_from_path(value.get("path"))
                return ImagePayload(media_type=media_type, data=bytes(raw_bytes))
            path = value.get("path")
            if path:
                image = Image.open(path)
                return self.encode_image(image)
        if isinstance(value, Image.Image):
            image = value.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return ImagePayload(media_type="image/png", data=buffer.getvalue())
        if isinstance(value, bytes):
            return ImagePayload(media_type="image/png", data=value)
        raise TypeError(f"Unsupported image payload: {type(value)!r}")

    def media_type_from_path(self, path: str | None) -> str:
        if not path:
            return "image/png"
        suffix = Path(path).suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        return "image/png"

    def cache_images(
        self,
        cache_dir: Path,
        sample_id: str,
        images: list[ImagePayload],
    ) -> list[Path]:
        digest = blake2b(sample_id.encode("utf-8"), digest_size=8).hexdigest()
        stem = self.safe_filename(sample_id)[:48]
        paths: list[Path] = []
        for index, image in enumerate(images):
            suffix = self.suffix_from_media_type(image.media_type)
            path = cache_dir / f"{stem}-{digest}-{index}{suffix}"
            path.write_bytes(image.data)
            paths.append(path)
        return paths

    def safe_filename(self, value: str) -> str:
        return (
            "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value).strip(
                "_"
            )
            or "sample"
        )

    def suffix_from_media_type(self, media_type: str) -> str:
        return {
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/png": ".png",
        }.get(media_type, ".png")

    @abstractmethod
    def build_sample_id(self, split: str, row_index: int, row: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_images(self, row: dict[str, Any]) -> list[ImagePayload]:
        raise NotImplementedError

    def extract_source_doc_id(self, row: dict[str, Any]) -> str | None:
        value = row.get("doc_id")
        return str(value) if value is not None else None

    def extract_data_source(self, row: dict[str, Any]) -> str | None:
        value = row.get("data_source")
        return str(value) if value is not None else None
