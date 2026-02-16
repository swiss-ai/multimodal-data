import glob
import logging
import os
import sys

import cv2
import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata


def cv2_decoder(key, data):
    extension = key.split(".")[-1].lower()
    if extension not in ["jpg", "jpeg", "png"]:
        return None
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def collate_fn(batch):
    return batch


class LAIONAestheticsAdapter(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        decode_workers: int = 0,
    ):
        self.data_dir = data_dir
        self.decode_workers = decode_workers

        shard_list = glob.glob(os.path.join(self.data_dir, "*.tar"))
        self.dataset = (
            wds.WebDataset(shard_list, shardshuffle=False)  # type:ignore
            .decode("pil")
            # .decode(cv2_decoder)
            .to_tuple("__key__", "jpg", "json")
        )

    @property
    def id(self):
        return "laion-aesthetics"

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        current_id = 0

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")
        dataset_stream = self.dataset

        loader = DataLoader(
            dataset_stream,
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

                item = batch_data[i]
                img = item[1]
                meta = item[2]

                m = SampleMetadata(
                    dataset_id=self.id,
                    sample_id=current_id,
                    data={
                        "dataset_id": self.id,
                        "meta": meta,
                    },
                )

                output_batch.append(ImageSample(image=img, meta=m))

                current_id += 1

            if output_batch:
                yield output_batch

            if current_id % 100000 < batch_len:
                logger.info(f"Yielded {current_id} samples so far...")

        logger.info("Finished streaming.")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    a = LAIONAestheticsAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/vision-datasets/LAION-Aesthetics",
    )

    for batch in a.stream(logger=logger, skip=0, batch_size=1000):
        # for b in batch:
        #     print("obtained sample:", b.meta.sample_id, b.image.size)
        # break
        print("obtained batch of size:", len(batch))
