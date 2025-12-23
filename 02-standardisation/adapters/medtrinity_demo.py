from datasets import load_dataset

from pipeline import BaseDataset, ImageSample, SampleMetadata


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

    def stream(self, skip: int | None = None):
        dataset = self.dataset
        start_idx = skip or 0

        if skip:
            dataset = dataset.skip(skip)  # type: ignore

        for idx, row in enumerate(dataset, start=start_idx):
            if idx == 3000:
                break  # TODO: remove limit

            yield ImageSample(
                image=row["image"],
                meta=SampleMetadata(
                    dataset_id=self.id,
                    sample_id=idx,
                    data={
                        "license": "CC BY-NC-SA 4.0",
                        "language": "en",
                    },
                ),
            )
