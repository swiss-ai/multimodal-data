from datasets import load_dataset

from src.adapter import BaseAdapter
from src.sample import ImageTextSample, SampleMetadata


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

    def stream(self):
        for idx, row in enumerate(self.dataset):
            if idx == 3000:
                break

            yield ImageTextSample(
                image=row["image"],
                text=row["caption"],
                meta=SampleMetadata(
                    dataset_id=self.name,
                    sample_id=str(idx),
                    data={
                        "license_type": "CC BY-NC-SA 4.0",
                        "language": "en",
                    },
                ),
            )
