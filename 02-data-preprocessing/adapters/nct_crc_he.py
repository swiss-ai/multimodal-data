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

# We use the colour-normalised training set + validation set.
# The NONORM variant is skipped to avoid duplication.
INNER_ZIPS = ["NCT-CRC-HE-100K.zip", "CRC-VAL-HE-7K.zip"]
EXTRACT_MARKER = ".extracted"

CLASS_CAPTIONS = {
    "ADI": "A histopathology image of colorectal tissue showing adipose tissue.",
    "BACK": "A histopathology image showing background in a colorectal biopsy slide.",
    "DEB": "A histopathology image of colorectal tissue showing cellular debris.",
    "LYM": "A histopathology image of colorectal tissue showing lymphocytic infiltrate.",
    "MUC": "A histopathology image of colorectal tissue showing mucus.",
    "MUS": "A histopathology image of colorectal tissue showing smooth muscle.",
    "NORM": "A histopathology image showing normal colorectal mucosa.",
    "STR": "A histopathology image of colorectal tissue showing cancer-associated stroma.",
    "TUM": "A histopathology image of colorectal tissue showing colorectal cancer epithelium.",
}


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


def _extract_if_needed(zip_path: str, extract_dir: str, logger) -> None:
    """Extract the two relevant inner zips from the outer zip to extract_dir.

    Each inner zip is first extracted as a file to extract_dir, then its
    contents are unpacked there, and the zip file is removed.  A marker file
    prevents re-extraction on subsequent runs.

    Resulting layout:
        extract_dir/NCT-CRC-HE-100K/{class}/*.tif
        extract_dir/CRC-VAL-HE-7K/{class}/*.tif
    """
    marker = os.path.join(extract_dir, EXTRACT_MARKER)
    if os.path.exists(marker):
        return

    logger.info(f"[nct_crc_he] Extracting {zip_path} -> {extract_dir}")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path) as outer:
        for inner_name in INNER_ZIPS:
            logger.info(f"[nct_crc_he]   Saving inner zip: {inner_name}")
            outer.extract(inner_name, extract_dir)
            inner_zip_path = os.path.join(extract_dir, inner_name)

            logger.info(f"[nct_crc_he]   Unpacking {inner_name}")
            with zipfile.ZipFile(inner_zip_path) as inner:
                inner.extractall(extract_dir)

            os.remove(inner_zip_path)
            logger.info(f"[nct_crc_he]   Done: {inner_name}")

    with open(marker, "w") as f:
        f.write("ok")
    logger.info("[nct_crc_he] Extraction complete.")


class NctCrcHeAdapter(BaseDataset):
    """Adapter for the NCT-CRC-HE-100K colorectal histopathology dataset.

    ~107K H&E-stained tissue patches (224x224 px) from 9 tissue classes,
    combining the normalised training set (100K) and validation set (7K).

    The dataset lives inside a doubly-nested zip:
        NCT-CRC-HE-100K.zip          <- outer zip
            NCT-CRC-HE-100K.zip      <- normalised training (used)
            CRC-VAL-HE-7K.zip        <- validation (used)
            NCT-CRC-HE-100K-NONORM.zip  <- skipped

    On first use the inner zips are extracted to extract_dir (set this to
    something like <raw_dir>/NCT-CRC-HE-100K_uncompressed).  Extraction is
    skipped on subsequent runs via a marker file.
    """

    def __init__(
        self,
        zip_path: str,
        extract_dir: str,
        dataset_id: str = "nct_crc_he",
        decode_workers: int = 64,
    ):
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self._id = dataset_id
        self.decode_workers = decode_workers

    @property
    def id(self) -> str:
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        _extract_if_needed(self.zip_path, self.extract_dir, logger)

        # Collect all image paths, sorted for deterministic ordering
        image_files: list[tuple[str, str]] = []
        for root, _dirs, files in os.walk(self.extract_dir):
            class_name = os.path.basename(root)
            caption = CLASS_CAPTIONS.get(class_name)
            if caption is None:
                continue
            for fname in sorted(files):
                if fname.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                    image_files.append((os.path.join(root, fname), caption))
        image_files.sort()

        logger.info(f"[{self.id}] Found {len(image_files):,} images in {self.extract_dir}")

        skip = skip or 0
        counter = 0
        pending: list[tuple[int, bytes, str]] = []

        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
            for img_path, caption in image_files:
                if counter < skip:
                    counter += 1
                    continue

                try:
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                except Exception:
                    logger.warning(f"[{self.id}] Failed to read {img_path}")
                    continue

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
    logger = logging.getLogger("nct_crc_he")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    ZIP_PATH = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/apertus/NCT-CRC-HE-100K.zip"
    EXTRACT_DIR = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/apertus/NCT-CRC-HE-100K_uncompressed"

    adapter = NctCrcHeAdapter(zip_path=ZIP_PATH, extract_dir=EXTRACT_DIR)
    for batch in adapter.stream(logger=logger, batch_size=4):
        for s in batch:
            print(f"  [{s.meta.sample_id}] {s.text}")
        break
