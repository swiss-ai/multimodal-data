import io
from enum import Enum, auto
from typing import Any, Dict, Union

import msgspec
from PIL.Image import Image


class SampleMetadata(msgspec.Struct):
    dataset_id: str
    sample_id: str
    data: Dict[str, Any]


class TextSample(msgspec.Struct):
    text: str
    meta: SampleMetadata

    def export_content(self) -> Dict[str, bytes]:
        if self.text:
            return {"txt": self.text.encode("utf-8")}
        return {}


class ImageSample(msgspec.Struct):
    image: Image
    meta: SampleMetadata

    def export_content(self) -> Dict[str, bytes]:
        if self.image and self.image.format:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format=self.image.format)
            return {self.image.format: img_byte_arr.getvalue()}
        return {}


class ImageTextSample(msgspec.Struct):
    image: Image
    text: str
    meta: SampleMetadata

    def export_content(self) -> Dict[str, bytes]:
        content = {}
        if self.image:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format=self.image.format)
            content[self.image.format] = img_byte_arr.getvalue()
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
