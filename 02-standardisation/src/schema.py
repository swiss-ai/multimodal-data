import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict

from PIL.Image import Image


@dataclass
class SampleMetadata:
    dataset_id: str
    sample_id: str
    data: Dict[str, Any]


@dataclass
class RawSample(ABC):
    """
    A raw data sample with methods to export its content for storage.
    """

    meta: SampleMetadata

    @abstractmethod
    def export_content(self) -> Dict[str, bytes]:
        """Exports the sample's content as a mapping of file extensions to byte data."""
        raise NotImplementedError


@dataclass
class TextSample(RawSample):
    """A sample containing text-only data."""

    text: str

    def export_content(self) -> Dict[str, bytes]:
        content = {}
        if self.text:
            content["txt"] = self.text.encode("utf-8")
        return content


@dataclass
class ImageSample(RawSample):
    """A sample containing image-only data."""

    image: Image

    def export_content(self) -> Dict[str, bytes]:
        content = {}
        if self.image:
            img_byte_arr = io.BytesIO()
            self.image.save(img_byte_arr, format=self.image.format)
            content[self.image.format] = img_byte_arr.getvalue()
        return content


@dataclass
class ImageTextSample(RawSample):
    """A sample containing both image and text data."""

    image: Image
    text: str

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
