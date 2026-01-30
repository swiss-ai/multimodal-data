import logging
import os
import sys

from datasets import load_dataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, ImageTextSample, SampleMetadata


def collate_fn(batch):
    return batch


class OpenPMC18mAdapter(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        decode_workers: int = 0,
        image_only: bool = True,
    ):
        self.data_dir = data_dir
        self.decode_workers = decode_workers
        self.image_only = image_only

        ds = load_dataset("webdataset", data_dir=data_dir, streaming=True)
        self.dataset = ds["train"]

    @property
    def id(self):
        return "open_pmc_18m"

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        current_id = 0

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")
        dataset_stream = self.dataset

        loader = DataLoader(
            dataset_stream,  # type: ignore
            batch_size=batch_size,
            num_workers=self.decode_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )

        for batch_data in loader:
            batch_len = len(batch_data)

            if current_id + batch_len <= skip:
                current_id += batch_len
                continue

            output_batch = []

            for i in range(batch_len):
                if current_id < skip:
                    current_id += 1
                    continue

                m = SampleMetadata(
                    dataset_id=self.id,
                    sample_id=current_id,
                    data={"dataset_id": self.id},
                )

                item = batch_data[i]
                img = item["jpg"]

                if self.image_only:
                    output_batch.append(ImageSample(image=img, meta=m))
                else:
                    text = item["json"]["caption"]
                    output_batch.append(ImageTextSample(image=img, text=text, meta=m))

                current_id += 1

            if output_batch:
                yield output_batch

            if current_id % 100000 < batch_len:
                logger.info(f"Yielded {current_id} samples so far...")

        logger.info("Finished streaming.")


if __name__ == "__main__":
    logger = logging.getLogger("open_pmc_18m")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    a = OpenPMC18mAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/open-pmc-18m",
    )

    for batch in a.stream(logger=logger, skip=5, batch_size=1000):
        # for b in batch:
        #     print("obtained sample:", b.meta.sample_id, b.image.size)
        # break

        print("obtained batch of size:", len(batch))
