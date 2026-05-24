import logging
import os
import sys

import PIL.Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, SampleMetadata

SPLIT_DIRS = [
    "Test_set",
    "Training_set",
    "Validation_set",
]


class RFMiD2Adapter(BaseDataset):
    def __init__(self, data_dir):
        dir_paths = [os.path.join(data_dir, d) for d in SPLIT_DIRS]
        image_paths = []
        for dir_path in dir_paths:
            for file in os.listdir(dir_path):
                if not file.endswith(".jpg"):
                    continue
                image_paths.append(os.path.join(dir_path, file))
        image_paths.sort()
        self.image_paths = image_paths
        self.data_dir = data_dir

    @property
    def id(self):
        return "rfmid2"

    def stream(self, logger, skip=None, batch_size=1):
        skip = skip or 0
        paths = self.image_paths[skip:]

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")

        pending = []
        for sample_id, path in enumerate(paths, start=skip):
            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=sample_id,
                data={"dataset_id": self.id, "path": path},
            )

            with open(path, "rb") as f:
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

    a = RFMiD2Adapter(
        data_dir=os.getenv("RFMID2_DATA_DIR", "/path/to/data"),
    )

    total = 0
    for batch in a.stream(logger=logger, skip=0, batch_size=100):
        total += len(batch)
        print("Processed batch of size:", len(batch))
    print("Total samples:", total)
