import io
from enum import Enum, auto
from typing import Any, Dict
from abc import ABC, abstractmethod

from PIL.Image import Image


class RawSample(ABC):
    """
    A raw data sample with methods to export its content for storage.
    """

    @abstractmethod
    def export_content(self) -> Dict[str, bytes]:
        """Exports the sample's content to a file extension to byte data mapping."""
        raise NotImplementedError


class SampleMetadata:
    dataset_id: str
    sample_id: str
    data: Dict[str, Any]


class TextSample(RawSample):
    """A sample containing text-only data."""

    text: str
    meta: SampleMetadata

    def export_content(self) -> Dict[str, bytes]:
        content = {}
        if self.text:
            content["txt"] = self.text.encode("utf-8")
        return content


class ImageSample(RawSample):
    """A sample containing image-only data."""

    image: Image
    meta: SampleMetadata

    def export_content(self) -> Dict[str, bytes]:
        content = {}
        if self.image:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format=self.image.format)
            content[self.image.format] = img_byte_arr.getvalue()
        return content


class ImageTextSample(RawSample):
    """A sample containing both image and text data."""

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


class SampleType(Enum):
    TEXT = auto()
    IMAGE = auto()
    IMAGE_TEXT = auto()
