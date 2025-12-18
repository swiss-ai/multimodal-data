from datasets import load_dataset
from PIL import Image as PILImage
from typing import Iterator

from src.adapter.base import BaseAdapter
from src.schema.image_format import ImageFormat
from src.schema.language import Language
from src.schema.license import License
from src.schema.sample import (
    ImageTextSample,
    ImageTextSampleMetadata,
    RawSample,
)


class MedtrinityDemoAdapter(BaseAdapter):
    def __init__(self):
        self.dataset = load_dataset(
            "UCSC-VLAA/MedTrinity-25M",
            "25M_demo",
            split="train",
            streaming=True,
        )

    @property
    def name(self):
        return "medtrinity_demo"

    def stream(self) -> Iterator[ImageTextSample]:
        for idx, row in enumerate(self.dataset):
            if idx == 2000:
                break

            pil_image: PILImage.Image = row["image"]
            caption: str = row.get("caption", "")

            fmt_enum = ImageFormat.PNG
            if pil_image.format == "JPEG":
                fmt_enum = ImageFormat.JPEG
            else:
                raise ValueError(f"unexpected format: {pil_image.format}")

            meta = ImageTextSampleMetadata(
                dataset_name=self.name,
                sample_id=str(idx),
                license_type=License.CC_BY_NC_SA_4_0.name,  # type: ignore
                text_language=Language.ENGLISH,
                text_length=len(caption),
                image_resolution=pil_image.size,
                image_format=fmt_enum,
                properties={},
            )

            yield ImageTextSample(
                image=pil_image.tobytes(),
                text=caption,
                meta=meta,
            )

    def hydrate(self, sample: RawSample) -> RawSample:
        return sample  # already hydrated
