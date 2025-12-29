import io
from PIL import Image
from datasets import load_from_disk

from pipeline import BaseDataset, ImageSample, SampleMetadata


class MeditronImageAdapter(BaseDataset):
    def __init__(self, dataset_id: str, data_dir: str):
        self.data_dir = data_dir
        self._id = dataset_id
        self.dataset = load_from_disk(data_dir)

    @property
    def id(self):
        return self._id

    def stream(self, logger, skip: int | None = None):
        skip = skip or 0
        counter = 0

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")
        if skip > 0:
            logger.info(f"Skipping first {skip} images.")

        for modality in self.dataset["modalities"]:
            for item in modality:
                try:
                    mod_type = item["type"]
                    if mod_type != "image":
                        continue
                    if counter < skip:
                        counter += 1
                        continue

                    image_bytes = item["value"]["bytes"]
                    if not image_bytes:
                        continue

                    img = Image.open(io.BytesIO(image_bytes))
                    if img.mode in ("P", "RGBA", "LA"):
                        img = img.convert("RGBA")
                    img = img.convert("RGB")

                    meta = SampleMetadata(
                        dataset_id=self.id,
                        sample_id=counter,
                        data={},
                    )

                    yield ImageSample(image=img, meta=meta)

                    counter += 1
                    if counter % 8000 == 0:
                        logger.info(f"Streamed {counter} samples so far...")

                except Exception as e:
                    logger.warning(f"Error processing item: {e}")

        logger.info("Finished streaming.")
