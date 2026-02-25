import io
import logging
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, MultiImageTextSample, Sample, SampleMetadata

# Images live at: {Training|Testing}/{class}/{name}.jpg
CLASS_CAPTIONS = {
    "glioma": "An MRI scan of the brain showing a glioma tumor.",
    "meningioma": "An MRI scan of the brain showing a meningioma.",
    "notumor": "An MRI scan of the brain showing no tumor.",
    "pituitary": "An MRI scan of the brain showing a pituitary tumor.",
}


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


class BrainTumorMRIAdapter(BaseDataset):
    """Adapter for the Brain Tumor MRI Dataset (7,200 images).

    Four classes: glioma, meningioma, notumor, pituitary.
    Both Training and Testing splits are included.

    Archive layout:
        brain-tumor-mri-dataset.zip
            Training/{class}/{name}.jpg
            Testing/{class}/{name}.jpg
    """

    def __init__(
        self,
        zip_path: str,
        dataset_id: str = "brain_tumor_mri",
        decode_workers: int = 64,
    ):
        self.zip_path = zip_path
        self._id = dataset_id
        self.decode_workers = decode_workers

    @property
    def id(self) -> str:
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0
        pending: list[tuple[int, bytes, str]] = []

        logger.info(f"[{self.id}] Streaming from {self.zip_path}")

        with (
            zipfile.ZipFile(self.zip_path, "r") as zf,
            ThreadPoolExecutor(max_workers=self.decode_workers) as executor,
        ):
            for zi in zf.infolist():
                if not zi.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Expect: {Training|Testing}/{class}/{name}.jpg
                parts = zi.filename.split("/")
                if len(parts) < 3 or parts[0] not in ("Training", "Testing"):
                    continue
                class_name = parts[1]
                caption = CLASS_CAPTIONS.get(class_name)
                if not caption:
                    continue

                if counter < skip:
                    counter += 1
                    continue

                img_bytes = zf.read(zi)
                pending.append((counter, img_bytes, caption))
                counter += 1

                if len(pending) >= batch_size:
                    yield self._decode_batch(executor, pending, logger)
                    pending = []

            if pending:
                yield self._decode_batch(executor, pending, logger)

        logger.info(f"[{self.id}] Finished streaming ({counter:,} total).")

    def _decode_batch(
        self,
        executor: ThreadPoolExecutor,
        pending: list[tuple[int, bytes, str]],
        logger,
    ) -> list[Sample]:
        sample_ids = [p[0] for p in pending]
        image_bytes_list = [p[1] for p in pending]
        captions = [p[2] for p in pending]

        pil_images = list(executor.map(_decode_image, image_bytes_list))

        batch = []
        for sample_id, img, caption in zip(sample_ids, pil_images, captions):
            if img is None:
                logger.warning(f"[{self.id}] Failed to decode image at index {sample_id}")
                continue
            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=sample_id,
                data={"dataset_id": self.id},
            )
            batch.append(MultiImageTextSample(images=[img], text=caption, meta=m))
        return batch


if __name__ == "__main__":
    logger = logging.getLogger("brain_tumor_mri")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    ZIP_PATH = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/apertus/brain-tumor-mri-dataset.zip"
    adapter = BrainTumorMRIAdapter(zip_path=ZIP_PATH)
    for batch in adapter.stream(logger=logger, batch_size=4):
        for s in batch:
            print(f"  [{s.meta.sample_id}] {s.text}")
        break
