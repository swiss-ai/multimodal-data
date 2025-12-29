import io
from concurrent.futures import ThreadPoolExecutor

from datasets import load_from_disk
from PIL import Image

from pipeline import BaseDataset, ImageSample, Sample, SampleMetadata


def _decode_image(image_bytes: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


class MeditronImageAdapter(BaseDataset):
    def __init__(self, dataset_id: str, data_dir: str, decode_workers: int = 64):
        self.data_dir = data_dir
        self._id = dataset_id
        self.decode_workers = decode_workers
        self.dataset = load_from_disk(data_dir)

    @property
    def id(self):
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")
        if skip > 0:
            logger.info(f"Skipping first {skip} images.")

        pending = []  # list of (sample_id, image_bytes)

        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
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

                        pending.append((counter, image_bytes))
                        counter += 1

                        if len(pending) >= batch_size:
                            batch = self._decode_batch(executor, pending, logger)
                            if batch:
                                yield batch
                            pending = []

                        if counter % 8000 == 0:
                            logger.info(f"Streamed {counter} samples so far...")

                    except Exception as e:
                        logger.warning(f"Error processing item: {e}")

            # remaining
            if pending:
                batch = self._decode_batch(executor, pending, logger)
                if batch:
                    yield batch

        logger.info("Finished streaming.")

    def _decode_batch(
        self,
        executor: ThreadPoolExecutor,
        pending: list[tuple[int, bytes]],
        logger,
    ) -> list[Sample]:
        """Decode a batch of images in parallel."""
        sample_ids = [p[0] for p in pending]
        image_bytes_list = [p[1] for p in pending]

        decoded = list(executor.map(_decode_image, image_bytes_list))

        batch = []
        for sample_id, img in zip(sample_ids, decoded):
            if img is None:
                logger.warning(f"Failed to decode image for sample {sample_id}")
                continue
            batch.append(
                ImageSample(
                    image=img,
                    meta=SampleMetadata(
                        dataset_id=self.id,
                        sample_id=sample_id,
                        data={},
                    ),
                )
            )
        return batch
