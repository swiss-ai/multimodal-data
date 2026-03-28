import logging
import os
import sys
from zipfile import ZipFile

import PIL.Image
import polars as pl

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata

IMAGE_DIRECTORY = "ISIC_2024_Permissive_Training_Input"
METADATA_FILE = os.path.join(IMAGE_DIRECTORY, "metadata.csv")


class ISICAdapter(BaseDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.zipfile = ZipFile(os.path.join(data_dir, IMAGE_DIRECTORY + ".zip"))
        self.dataset = pl.read_csv(self.zipfile.open(METADATA_FILE).read())

    @property
    def id(self):
        return "isic"

    def stream(self, logger, skip=None, batch_size=1):
        skip = skip or 0
        df = self.dataset.slice(skip)

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")

        pending = []
        for sample_id, row in enumerate(df.iter_rows(named=True), start=skip):
            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=sample_id,
                data={
                    "dataset_id": self.id,
                    "patient_info": row,
                },
            )

            img_path = os.path.join(IMAGE_DIRECTORY, row["isic_id"] + ".jpg")
            with self.zipfile.open(img_path) as f:
                img = PIL.Image.open(f).convert("RGB")
            sample = ImageSample(meta=m, image=img)
            pending.append(sample)

            if len(pending) >= batch_size:
                yield pending
                pending = []

        if pending:
            yield pending

        logger.info("Finished streaming.")


if __name__ == "__main__":
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    a = ISICAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/isic",
    )

    total = 0
    for batch in a.stream(logger=logger, skip=0, batch_size=1000):
        total += len(batch)
        # for b in batch:
        #     print("obtained sample:", b.meta.sample_id, b.image.size)
        # break
        print("Processed batch of size:", len(batch))
    print("Total samples:", total)
