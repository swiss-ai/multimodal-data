import logging
import os
import sys

from datasets import load_dataset

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata

columns = [
    "image_1_path",
    "image_2_path",
    "image_3_path",
]


def collate_fn(batch):
    return batch


class SCINAdapter(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        decode_workers: int = 0,
    ):
        self.data_dir = data_dir
        self.decode_workers = decode_workers

        ds = load_dataset("parquet", data_dir=data_dir, streaming=True)
        self.dataset = ds.select_columns(columns)["train"]

    @property
    def id(self):
        return "scin"

    def stream(self, logger, skip=None, batch_size=1):
        skip = skip or 0
        current_id = skip

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")

        pending = []
        for sample in self.dataset.skip(skip):  # type: ignore
            for col in columns:
                img = sample[col]  # type: ignore
                if img is None:
                    continue

                m = SampleMetadata(
                    dataset_id=self.id,
                    sample_id=current_id,
                    data={"dataset_id": self.id},
                )

                pending.append(ImageSample(image=img, meta=m))
                current_id += 1

            if len(pending) >= batch_size:
                yield pending[:batch_size]
                pending = pending[batch_size:]

        if pending:
            yield pending

        logger.info("Finished streaming.")


if __name__ == "__main__":
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    a = SCINAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/scin",
    )

    total = 0
    for batch in a.stream(logger=logger, skip=0, batch_size=1000):
        total += len(batch)
        # for b in batch:
        #     print("obtained sample:", b.meta.sample_id, b.image.size)
        # break
        print("Processed batch of size:", len(batch))
    print("Total samples:", total)
