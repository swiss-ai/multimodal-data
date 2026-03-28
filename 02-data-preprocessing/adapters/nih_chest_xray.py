import csv
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

GENDER_MAP = {"M": "male", "F": "female"}


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


def _build_caption(row: dict) -> str:
    findings = row["Finding Labels"].strip()
    age = row["Patient Age"].strip()
    gender = GENDER_MAP.get(row["Patient Gender"].strip(), "unknown gender")
    view = row["View Position"].strip()
    if findings == "No Finding":
        return (
            f"A chest X-ray ({view} view) of a {age}-year-old {gender} patient "
            f"showing no significant pathological findings."
        )
    findings_text = findings.replace("|", ", ")
    return (
        f"A chest X-ray ({view} view) of a {age}-year-old {gender} patient "
        f"showing findings of {findings_text}."
    )


class NihChestXrayAdapter(BaseDataset):
    """Adapter for the NIH Chest X-ray Dataset (112,120 frontal-view images).

    Reads nih-chest-xrays.zip directly. Labels loaded from Data_Entry_2017.csv.
    Generates descriptive captions from finding labels and patient demographics.

    CSV fields used: Image Index, Finding Labels, Patient Age, Patient Gender,
    View Position.
    """

    def __init__(
        self,
        zip_path: str,
        dataset_id: str = "nih_chest_xray",
        decode_workers: int = 64,
    ):
        self.zip_path = zip_path
        self._id = dataset_id
        self.decode_workers = decode_workers

        # Build lookup: image filename -> caption
        self.lookup: dict[str, str] = {}
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open("Data_Entry_2017.csv") as f:
                reader = csv.DictReader(io.TextIOWrapper(f))
                for row in reader:
                    img_name = row["Image Index"].strip()
                    self.lookup[img_name] = _build_caption(row)

    @property
    def id(self) -> str:
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0
        pending: list[tuple[int, bytes, str]] = []

        logger.info(f"[{self.id}] Streaming from {self.zip_path} ({len(self.lookup):,} entries in CSV)")
        if skip > 0:
            logger.info(f"[{self.id}] Skipping first {skip} samples.")

        with (
            zipfile.ZipFile(self.zip_path, "r") as zf,
            ThreadPoolExecutor(max_workers=self.decode_workers) as executor,
        ):
            for zi in zf.infolist():
                if not zi.filename.lower().endswith(".png"):
                    continue
                filename = os.path.basename(zi.filename)
                caption = self.lookup.get(filename)
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

                if counter % 10000 == 0:
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
    logger = logging.getLogger("nih_chest_xray")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    ZIP_PATH = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/apertus/nih-chest-xrays.zip"
    adapter = NihChestXrayAdapter(zip_path=ZIP_PATH)
    for batch in adapter.stream(logger=logger, batch_size=4):
        for s in batch:
            assert isinstance(s, MultiImageTextSample)
            print(f"  [{s.meta.sample_id}] {s.text}")
        break
