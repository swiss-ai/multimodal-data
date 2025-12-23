import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import msgspec
from PIL import Image


@dataclass
class SampleMetadata:
    dataset_id: str
    sample_id: int
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sample(ABC):
    """Base class for all sample types."""

    meta: SampleMetadata

    @abstractmethod
    def serialize(self) -> bytes:
        """Serialize to bytes."""
        ...

    @staticmethod
    def deserialize(data: bytes) -> "Sample":
        """Deserialize from bytes."""
        msg = msgspec.msgpack.decode(data, type=SerializedSample)
        return msg.to_sample()


@dataclass
class TextSample(Sample):
    """Text-only sample."""

    text: str

    def serialize(self) -> bytes:
        return msgspec.msgpack.encode(
            SerializedSample(
                sample_type="text",
                dataset_id=self.meta.dataset_id,
                sample_id=self.meta.sample_id,
                meta_data=self.meta.data,
                text=self.text,
            )
        )


@dataclass
class ImageSample(Sample):
    """Image-only sample."""

    image: Image.Image

    def serialize(self) -> bytes:
        buf = io.BytesIO()

        # serialize image bytes to buffer
        fmt = self.image.format or "PNG"
        if fmt.upper() in ("JPEG", "JPG") and self.image.mode == "RGBA":
            # jpeg does not support alpha channel
            self.image.convert("RGB").save(buf, format=fmt)
        else:
            self.image.save(buf, format=fmt)

        return msgspec.msgpack.encode(
            SerializedSample(
                sample_type="image",
                dataset_id=self.meta.dataset_id,
                sample_id=self.meta.sample_id,
                meta_data=self.meta.data,
                image_bytes=buf.getvalue(),
                image_format=fmt,
            )
        )


@dataclass
class ImageTextSample(Sample):
    """Image + text sample."""

    image: Image.Image
    text: str

    def serialize(self) -> bytes:
        buf = io.BytesIO()

        # serialize image bytes to buffer
        fmt = self.image.format or "PNG"
        if fmt.upper() in ("JPEG", "JPG") and self.image.mode == "RGBA":
            # jpeg does not support alpha channel
            self.image.convert("RGB").save(buf, format=fmt)
        else:
            self.image.save(buf, format=fmt)

        return msgspec.msgpack.encode(
            SerializedSample(
                sample_type="image_text",
                dataset_id=self.meta.dataset_id,
                sample_id=self.meta.sample_id,
                meta_data=self.meta.data,
                text=self.text,
                image_bytes=buf.getvalue(),
                image_format=fmt,
            )
        )


class SerializedSample(msgspec.Struct):
    """Msgspec struct for serialization."""

    sample_type: str
    dataset_id: str
    sample_id: int
    meta_data: dict[str, Any]

    # sample-type-specific fields
    text: str | None = None
    image_bytes: bytes | None = None
    image_format: str | None = None

    def to_sample(self) -> Sample:
        meta = SampleMetadata(
            dataset_id=self.dataset_id,
            sample_id=self.sample_id,
            data=self.meta_data,
        )

        if self.sample_type == "text":
            assert self.text is not None
            return TextSample(meta=meta, text=self.text)

        elif self.sample_type == "image":
            assert self.image_bytes is not None
            image = Image.open(io.BytesIO(self.image_bytes))
            image.load()
            image.format = self.image_format
            return ImageSample(meta=meta, image=image)

        elif self.sample_type == "image_text":
            assert self.image_bytes is not None
            assert self.text is not None
            image = Image.open(io.BytesIO(self.image_bytes))
            image.load()
            image.format = self.image_format
            return ImageTextSample(meta=meta, image=image, text=self.text)

        raise ValueError(f"Unknown sample type: {self.sample_type}")
