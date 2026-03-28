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

# Images live at: COVID-19_Radiography_Dataset/{class}/images/{name}.png
ROOT_DIR = "COVID-19_Radiography_Dataset"

CLASS_CAPTIONS = {
    "COVID": "A chest X-ray image showing COVID-19 pneumonia.",
    "Lung_Opacity": "A chest X-ray image showing lung opacity.",
    "Normal": "A normal chest X-ray image showing no significant pulmonary abnormalities.",
    "Viral Pneumonia": "A chest X-ray image showing viral pneumonia.",
}


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


class CovidRadiographyAdapter(BaseDataset):
    """Adapter for the COVID-19 Radiography Database (21,165 chest X-rays).

    Four classes: COVID (3,616), Lung Opacity (6,012), Normal (10,192),
    Viral Pneumonia (1,345). Only the image files under each class's
    `images/` subdirectory are processed; mask files are skipped.

    Archive layout:
        covid19-radiography-database.zip
            COVID-19_Radiography_Dataset/{class}/images/{name}.png
            COVID-19_Radiography_Dataset/{class}/masks/{name}.png   <- skipped
    """

    def __init__(
        self,
        zip_path: str,
        dataset_id: str = "covid_radiography",
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
                if not zi.filename.lower().endswith(".png"):
                    continue
                # Expect: ROOT_DIR/{class}/images/{name}.png
                parts = zi.filename.split("/")
                if len(parts) < 4 or parts[2] != "images":
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

                if counter % 5000 == 0:
                    logger.info(f"[{self.id}] Streamed {counter:,} samples so far.")

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
    logger = logging.getLogger("covid_radiography")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    ZIP_PATH = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/apertus/covid19-radiography-database.zip"
    adapter = CovidRadiographyAdapter(zip_path=ZIP_PATH)
    for batch in adapter.stream(logger=logger, batch_size=4):
        for s in batch:
            print(f"  [{s.meta.sample_id}] {s.text}")
        break
