import logging
import os
import sys

import PIL.Image
import polars as pl

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata

METADATA_FILE = "Demographics of the participants.xlsx"
IMAGE_DIRECTORY = "Original UWF Image"


class UWFAdapter(BaseDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        metadata_path = os.path.join(data_dir, METADATA_FILE)
        self.dataset = pl.read_excel(metadata_path)

    @property
    def id(self):
        return "uwf"

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
                    "image_id": row["Image ID "],
                    "patient_id": row["Patient ID"],
                    "eye_category": row["Eye category"],
                    "age": row["Age"],
                    "sex": row["Sex"],
                    "diagnosis": row["Diagnosis"],
                },
            )

            img_path = os.path.join(
                self.data_dir,
                IMAGE_DIRECTORY,
                row["Diagnosis"],
                row["Image ID "],
            )

            with open(img_path, "rb") as f:
                img = PIL.Image.open(f).convert("RGB")
                assert img.size in [(3900, 3072), (2600, 2048)]
                img = img.resize((1300, 1024))
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

    a = UWFAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/uwf",
    )

    total = 0
    for batch in a.stream(logger=logger, skip=0, batch_size=100):
        total += len(batch)
        # for b in batch:
        #     print("obtained sample:", b.meta.sample_id, b.image.size)
        # break
        print("Processed batch of size:", len(batch))
    print("Total samples:", total)
