from enum import Enum, auto
from typing import Any, Dict, Optional, Union

import msgspec

from src.schema.image import ImageFormat
from src.schema.language import Language
from src.schema.license import License


class SampleMetadata(msgspec.Struct):
    dataset_name: str
    sample_id: str
    license_type: License
    properties: Dict[str, Any]


class TextSample(msgspec.Struct):
    text: Optional[str]
    language: Language
    meta: SampleMetadata


class ImageSample(msgspec.Struct):
    image: Optional[bytes]
    resolution: tuple[int, int]
    format: ImageFormat
    meta: SampleMetadata


class ImageTextSample(msgspec.Struct):
    text: Optional[str]
    language: Language
    image: Optional[bytes]
    resolution: tuple[int, int]
    format: ImageFormat
    meta: SampleMetadata


# class AudioSample(msgspec.Struct):
#     audio: bytes
#     duration_seconds: float
#     format: str
#     sample_rate: int
#     meta: SampleMetadata


class SampleType(Enum):
    TEXT = auto()
    IMAGE = auto()
    IMAGE_TEXT = auto()


SampleTypeMap = {
    SampleType.TEXT: TextSample,
    SampleType.IMAGE: ImageSample,
    SampleType.IMAGE_TEXT: ImageTextSample,
}

RawSample = Union[
    TextSample,
    ImageSample,
    ImageTextSample,
]
