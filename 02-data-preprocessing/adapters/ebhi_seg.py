import io
import logging
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, MultiImageTextSample, Sample, SampleMetadata

# Class names sorted longest-first to avoid prefix ambiguity during matching.
# Each filename encodes the class as a prefix: "{class}_{specimen_id}.png"
EBHI_CLASSES = sorted(
    [
        "High-grade IN",
        "Low-grade IN",
        "Serrated adenoma",
        "Adenocarcinoma",
        "Normal",
        "Polyp",
    ],
    key=len,
    reverse=True,
)

CLASS_CAPTIONS = {
    "Adenocarcinoma": (
        "A histopathology image of colorectal tissue showing adenocarcinoma."
    ),
    "High-grade IN": (
        "A histopathology image of colorectal tissue showing high-grade "
        "intraepithelial neoplasia."
    ),
    "Low-grade IN": (
        "A histopathology image of colorectal tissue showing low-grade "
        "intraepithelial neoplasia."
    ),
    "Normal": "A histopathology image of normal colorectal tissue.",
    "Polyp": "A histopathology image of colorectal tissue showing a polyp.",
    "Serrated adenoma": (
        "A histopathology image of colorectal tissue showing a serrated adenoma."
    ),
}


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


def _extract_class(filename: str) -> str | None:
    """Extract the class name from an EBHI-Seg image filename.

    Filenames follow the pattern "{class}_{specimen_id}.png", where class names
    may contain spaces (e.g. "Low-grade IN_GT2012400-1-400-004.png").
    We match against the known class list (longest-first to avoid prefix clashes).
    """
    stem = os.path.splitext(filename)[0]
    for cls in EBHI_CLASSES:
        if stem.startswith(cls + "_"):
            return cls
    return None


class EBHISegAdapter(BaseDataset):
    """Adapter for the EBHI-Seg colorectal histopathology dataset (2,228 images).

    Six classes: Adenocarcinoma (795), Low-grade IN (639), Polyp (474),
    High-grade IN (186), Normal (76), Serrated adenoma (58).

    Archive layout:
        EBHI-Seg.tar
            ds/img/{class}_{specimen_id}.png   <- images (class encoded in filename)
            ds/ann/{class}_{specimen_id}.png.json  <- segmentation labels (skipped)
            meta.json
    """

    def __init__(
        self,
        tar_path: str,
        dataset_id: str = "ebhi_seg",
        decode_workers: int = 32,
    ):
        self.tar_path = tar_path
        self._id = dataset_id
        self.decode_workers = decode_workers

    @property
    def id(self) -> str:
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0
        pending: list[tuple[int, bytes, str]] = []

        logger.info(f"[{self.id}] Streaming from {self.tar_path}")

        with (
            tarfile.open(self.tar_path, "r:") as tf,
            ThreadPoolExecutor(max_workers=self.decode_workers) as executor,
        ):
            for m in tf:
                if not m.isfile():
                    continue
                if not m.name.startswith("ds/img/"):
                    continue
                if not m.name.lower().endswith(".png"):
                    continue

                filename = os.path.basename(m.name)
                class_name = _extract_class(filename)
                if class_name is None:
                    logger.warning(f"[{self.id}] Could not extract class from {filename}")
                    continue
                caption = CLASS_CAPTIONS[class_name]

                if counter < skip:
                    counter += 1
                    continue

                f_obj = tf.extractfile(m)
                if f_obj is None:
                    continue
                try:
                    img_bytes = f_obj.read()
                except Exception:
                    logger.warning(f"[{self.id}] Failed to read {filename}")
                    continue

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
    logger = logging.getLogger("ebhi_seg")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    TAR_PATH = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/apertus/EBHI-Seg.tar"
    adapter = EBHISegAdapter(tar_path=TAR_PATH)
    for batch in adapter.stream(logger=logger, batch_size=4):
        for s in batch:
            print(f"  [{s.meta.sample_id}] {s.text}")
        break
