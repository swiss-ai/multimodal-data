from enum import Enum, auto
from typing import Any, Dict, Optional, Union

import msgspec

from src.schema.image_format import ImageFormat
from src.schema.language import Language
from src.schema.license import License


class SampleMetadata(msgspec.Struct):
    dataset_name: str
    sample_id: str
    license_type: License
    properties: Dict[str, Any]


class TextSampleMetadata(SampleMetadata):
    language: Language
    text_length: int


class ImageSampleMetadata(SampleMetadata):
    resolution: tuple[int, int]
    format: ImageFormat


class ImageTextSampleMetadata(SampleMetadata):
    text_language: Language
    text_length: int
    image_resolution: tuple[int, int]
    image_format: ImageFormat


class TextSample:
    text: Optional[str]
    meta: TextSampleMetadata

    def export_content(self) -> Dict[str, bytes]:
        if self.text:
            return {"txt": self.text.encode("utf-8")}
        return {}


class ImageSample:
    image: Optional[bytes]
    meta: ImageSampleMetadata

    def export_content(self) -> Dict[str, bytes]:
        if self.image:
            ext = self.meta.format.value
            return {ext: self.image}
        return {}


class ImageTextSample:
    image: Optional[bytes]
    text: Optional[str]
    meta: ImageTextSampleMetadata

    def export_content(self) -> Dict[str, bytes]:
        content = {}
        if self.image:
            content[self.meta.image_format.value] = self.image
        if self.text:
            content["txt"] = self.text.encode("utf-8")
        return content


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


RawSample = Union[
    TextSample,
    ImageSample,
    ImageTextSample,
]

SAMPLE_TYPE_MAP = {
    TextSample: SampleType.TEXT,
    ImageSample: SampleType.IMAGE,
    ImageTextSample: SampleType.IMAGE_TEXT,
}


def get_sample_type(sample: RawSample) -> SampleType:
    return SAMPLE_TYPE_MAP[type(sample)]
