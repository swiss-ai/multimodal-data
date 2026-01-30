import logging
import os
import sys

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata

subsets = [
    "pathmnist",
    "chestmnist",
    "dermamnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
]

splits = [
    "train_images",
    "val_images",
    "test_images",
]


class NumpyMapDataset(Dataset):
    def __init__(self, data_matrix):
        self.data = data_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return batch


def to_pil(img_array):
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    return Image.fromarray(img_array)


class MedMNISTAdapter(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        decode_workers: int = 0,
    ):
        self.data_dir = data_dir
        self.decode_workers = decode_workers

    @property
    def id(self):
        return "medmnist"

    def stream(self, logger, skip=None, batch_size=1):
        to_skip = skip or 0
        current_id = 0

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")

        for subset in subsets:
            npz_file = os.path.join(self.data_dir, f"{subset}_224.npz")
            data = np.load(npz_file)

            for split_name in splits:
                split_data = data[split_name]
                split_len = len(split_data)
                if to_skip >= split_len:
                    to_skip -= split_len
                    current_id += split_len
                    continue

                start_idx = to_skip
                current_id += start_idx
                to_skip = 0

                dataset_slice = NumpyMapDataset(split_data[start_idx:])
                loader = DataLoader(
                    dataset_slice,
                    batch_size=batch_size,
                    num_workers=self.decode_workers,
                    collate_fn=collate_fn,
                    pin_memory=False,
                )

                for batch_data in loader:
                    output_batch = []
                    batch_len = len(batch_data)

                    for i in range(batch_len):
                        m = SampleMetadata(
                            dataset_id=self.id,
                            sample_id=current_id,
                            data={
                                "dataset_id": self.id,
                                "subset": subset,
                                "split": split_name,
                            },
                        )

                        pil_image = to_pil(batch_data[i])
                        output_batch.append(ImageSample(image=pil_image, meta=m))
                        current_id += 1

                    if output_batch:
                        yield output_batch

            del data

        logger.info("Finished streaming.")


if __name__ == "__main__":
    from datetime import datetime

    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    a = MedMNISTAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/medmnist",
    )

    total = 0
    for batch in a.stream(logger=logger, skip=0, batch_size=1000):
        total += len(batch)
        print(f"[{datetime.now():%H:%M:%S}] {len(batch)}")
        # for b in batch:
        #     print("obtained sample:", b.meta.sample_id, b.image.shape)
        # break
    print("Total samples:", total)
