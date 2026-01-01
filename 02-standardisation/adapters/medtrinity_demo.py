import logging
import os
import sys

from datasets import load_dataset

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata


class MedtrinityDemoAdapter(BaseDataset):
    def __init__(self):
        self.dataset = load_dataset(
            "UCSC-VLAA/MedTrinity-25M",
            "25M_demo",
            split="train",
            streaming=True,
        )
        self.dataset = self.dataset.take(3000)  # type: ignore

    @property
    def id(self):
        return "medtrinity_demo"

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        if skip:
            logger.info(f"Skipping first {skip} samples.")
            _ = self.dataset = self.dataset.skip(skip)  # type: ignore

        batch = []
        for idx, row in enumerate(self.dataset, start=skip):
            batch.append(
                ImageSample(
                    image=row["image"],
                    meta=SampleMetadata(
                        dataset_id=self.id,
                        sample_id=idx,
                        data={"dataset_id": self.id},
                    ),
                )
            )

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


if __name__ == "__main__":
    logger = logging.getLogger("medtrinity_full")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info("initializing adapter...")

    a = MedtrinityDemoAdapter()

    logger.info("starting test stream...")
    for batch in a.stream(logger=logger, skip=0, batch_size=1000):
        for b in batch:
            print(
                "obtained sample:",
                b.meta.sample_id,
                b.image.size,  # type: ignore
                b.meta.data["dataset_id"],
            )
