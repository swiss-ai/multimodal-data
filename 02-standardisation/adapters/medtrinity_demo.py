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

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        dataset = self.dataset
        start_idx = skip or 0

        if skip:
            logger.info(f"Skipping first {skip} samples.")
            dataset = dataset.skip(skip)  # type: ignore

        batch = []
        for idx, row in enumerate(dataset, start=start_idx):
            batch.append(
                ImageSample(
                    image=row["image"],
                    meta=SampleMetadata(
                        dataset_id=self.id,
                        sample_id=idx,
                        data={"language": "en"},
                    ),
                )
            )
            if len(batch) >= batch_size:
                yield batch
                batch = []
            if idx % 2000 == 0:
                logger.info(f"Streamed {idx} samples so far...")

        if batch:
            yield batch
