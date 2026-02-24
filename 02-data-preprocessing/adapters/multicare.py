import logging
import os
import sys

from datasets import load_dataset

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, MultiImageTextSample, SampleMetadata


class MultiCaReAdapter(BaseDataset):
    """Adapter for MultiCaRe parquet datasets (multicare-case-images, multicare-images).

    Both datasets have 'image' and 'caption' columns in HuggingFace parquet format.
    """

    def __init__(
        self,
        data_dir: str,
        dataset_id: str = "multicare",
    ):
        self.data_dir = data_dir
        self._id = dataset_id

        ds = load_dataset("parquet", data_dir=data_dir, streaming=True)
        self.dataset = ds["train"]

    @property
    def id(self):
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        current_id = 0

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")

        pending = []
        for sample in self.dataset:
            if current_id < skip:
                current_id += 1
                continue

            img = sample["image"]
            caption = sample.get("caption", "")

            if img is None:
                current_id += 1
                continue

            if not caption:
                current_id += 1
                continue

            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=current_id,
                data={"dataset_id": self.id},
            )

            pending.append(MultiImageTextSample(images=[img], text=caption, meta=m))
            current_id += 1

            if len(pending) >= batch_size:
                yield pending
                pending = []

            if current_id % 10000 == 0:
                logger.info(f"Streamed {current_id} samples so far...")

        if pending:
            yield pending

        logger.info("Finished streaming.")


if __name__ == "__main__":
    logger = logging.getLogger("multicare")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    a = MultiCaReAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/multicare-case-images/data",
        dataset_id="multicare_case_images",
    )

    total = 0
    for batch in a.stream(logger=logger, skip=0, batch_size=100):
        total += len(batch)
        for b in batch:
            print(
                "obtained sample:",
                b.meta.sample_id,
                b.image.size,
                b.text[:50],
            )
        if total > 500:
            break
    print("Total samples:", total)
