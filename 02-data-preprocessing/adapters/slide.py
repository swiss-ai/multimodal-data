import io
import logging
import os
import sys
import zipfile

import PIL.Image
from torch.utils.data import DataLoader, Dataset, Subset

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata


class SLIDEDataset(Dataset):
    def __init__(self, data_dir, img_size=(768, 584)):
        self.dir_path = data_dir
        self.img_size = img_size

        zip_files = ["test.zip", "train.zip", "val.zip"]
        zip_paths = [os.path.join(data_dir, zip_file) for zip_file in zip_files]

        self.zip_handles = {}
        self.data_map = []
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, "r") as archive:
                self.data_map.extend(
                    [(zip_path, f) for f in archive.namelist() if f.endswith(".jpg")]
                )

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        zip_path, file_name = self.data_map[idx]
        if zip_path not in self.zip_handles:
            self.zip_handles[zip_path] = zipfile.ZipFile(zip_path, "r")
        with self.zip_handles[zip_path].open(file_name) as file:
            img_data = file.read()
        image = PIL.Image.open(io.BytesIO(img_data))
        assert image.size == (2559, 1957)
        image = image.resize(self.img_size)
        return image

    def __del__(self):
        for z in self.zip_handles.values():
            z.close()

    @staticmethod
    def collate_fn(batch):
        return batch


class SLIDEAdapter(BaseDataset):
    def __init__(self, data_dir, decode_workers=0):
        self.data_dir = data_dir
        self.decode_workers = decode_workers
        self.dataset = SLIDEDataset(data_dir)

    @property
    def id(self):
        return "slide"

    def stream(self, logger, skip=None, batch_size=1):
        skip = skip or 0
        valid_indices = range(skip, len(self.dataset))
        dataset = Subset(self.dataset, valid_indices)
        current_id = skip

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=SLIDEDataset.collate_fn,
            num_workers=self.decode_workers,
        )

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")

        for batch in dataloader:
            output_batch = []
            for img in batch:
                m = SampleMetadata(
                    dataset_id=self.id,
                    sample_id=current_id,
                    data={"dataset_id": self.id},
                )
                output_batch.append(ImageSample(image=img, meta=m))
                current_id += 1

            yield output_batch

        logger.info("Finished streaming.")


if __name__ == "__main__":
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    a = SLIDEAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/slide",
        decode_workers=3,
    )

    total = 0
    for batch in a.stream(logger=logger, skip=1998, batch_size=100):
        total += len(batch)
        # for b in batch:
        #     print("obtained sample:", b.meta.sample_id, b.image.size)
        # break
        print("Processed batch of size:", len(batch))
    print("Total samples:", total)
