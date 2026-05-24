from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ImagePayload:
    media_type: str
    data: bytes


@dataclass(slots=True)
class ManifestRecord:
    sample_id: str
    split: str
    row_index: int
    cached_image_paths: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SourceSample:
    sample_id: str
    split: str
    row_index: int
    images: list[ImagePayload]
    metadata: dict[str, Any]


@dataclass(slots=True)
class CandidateExample:
    sample_id: str
    source_sample_id: str
    source_split: str
    source_row_index: int
    task_type: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_sample_id": self.source_sample_id,
            "source_split": self.source_split,
            "source_row_index": self.source_row_index,
            "task_type": self.task_type,
            "messages": self.messages,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class CuratedExample:
    sample_id: str
    source_sample_id: str
    source_split: str
    source_row_index: int
    task_type: str
    messages: list[dict[str, str]]
    quality_score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_sample_id": self.source_sample_id,
            "source_split": self.source_split,
            "source_row_index": self.source_row_index,
            "task_type": self.task_type,
            "messages": self.messages,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }
