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

    def stream(self, from_id=None):
        for idx, row in enumerate(self.dataset):
            if idx == 3000:
                break  # TODO: remove limit

            sample_id = str(idx)

            if from_id and sample_id <= from_id:
                from_id = None
                continue

            yield ImageSample(
                image=row["image"],
                meta=SampleMetadata(
                    dataset_id=self.id,
                    sample_id=sample_id,
                    data={
                        "license": "CC BY-NC-SA 4.0",
                        "language": "en",
                    },
                ),
            )
