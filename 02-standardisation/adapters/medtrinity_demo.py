from datasets import load_dataset

from src.base import BaseDataset
from src.schema import ImageSample, SampleMetadata


class MedtrinityDemoAdapter(BaseDataset):
    def __init__(self):
        self.dataset = load_dataset(
            "UCSC-VLAA/MedTrinity-25M",
            "25M_demo",
            split="train",
            streaming=True,
        )

    @property
    def id(self):
        return "medtrinity_demo"

    def __iter__(self):
        for idx, row in enumerate(self.dataset):
            if idx == 3000:
                break

            yield ImageSample(
                image=row["image"],
                meta=SampleMetadata(
                    dataset_id=self.id,
                    sample_id=str(idx),
                    data={
                        "license_type": "CC BY-NC-SA 4.0",
                        "language": "en",
                    },
                ),
            )
