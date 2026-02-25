import io
import json
import logging
import os
import pickle
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, MultiImageTextSample, Sample, SampleMetadata

JSONL_NAME = "release/s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl"


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


def _build_lookup(tar_path: str) -> dict[str, str]:
    """Stream through tar to find and parse the JSONL metadata file.

    Returns {image_filename: caption} where filename = "{pdf_hash}_{fig_uri}".
    The JSONL entry is near the end of the archive, so the full tar is scanned.
    """
    lookup: dict[str, str] = {}
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if m.name != JSONL_NAME:
                continue
            f_obj = tf.extractfile(m)
            if f_obj is None:
                break
            for line in f_obj:
                obj = json.loads(line)
                pdf_hash = obj.get("pdf_hash", "")
                fig_uri = obj.get("fig_uri", "")
                caption = obj.get("s2_caption") or obj.get("s2orc_caption", "")
                if pdf_hash and fig_uri and caption:
                    lookup[f"{pdf_hash}_{fig_uri}"] = caption
            break
    return lookup


class MediCaTAdapter(BaseDataset):
    """Adapter for MediCaT (Medical Image Captions and Triples) dataset.

    217K medical figures extracted from open-access papers, each paired with
    a figure caption from Semantic Scholar / S2ORC.

    Archive layout:
        medicat.tar.gz
            release/figures/{pdf_hash}_{fig_uri}   <- images
            release/s2_full_figures_oa_nonroco_combined_medical_top4_public.jsonl
    """

    def __init__(
        self,
        tar_path: str,
        cache_file: str,
        dataset_id: str = "medicat",
        decode_workers: int = 64,
    ):
        self.tar_path = tar_path
        self._id = dataset_id
        self.decode_workers = decode_workers

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.lookup: dict[str, str] = pickle.load(f)
        else:
            self.lookup = _build_lookup(tar_path)
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.lookup, f)

    @property
    def id(self) -> str:
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0
        pending: list[tuple[int, bytes, str]] = []

        logger.info(f"[{self.id}] Streaming from {self.tar_path} ({len(self.lookup):,} captions loaded)")
        if skip > 0:
            logger.info(f"[{self.id}] Skipping first {skip} samples.")

        with (
            tarfile.open(self.tar_path, "r|gz") as tf,
            ThreadPoolExecutor(max_workers=self.decode_workers) as executor,
        ):
            for m in tf:
                if not m.isfile():
                    continue
                filename = os.path.basename(m.name)
                if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue

                caption = self.lookup.get(filename)
                if not caption:
                    continue

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
    logger = logging.getLogger("medicat")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    TAR_PATH = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/apertus/medicat.tar.gz"
    CACHE = "/capstor/scratch/cscs/tchu/.cache/medicat_metadata.pkl"

    adapter = MediCaTAdapter(tar_path=TAR_PATH, cache_file=CACHE)
    for batch in adapter.stream(logger=logger, batch_size=4):
        for s in batch:
            assert isinstance(s, MultiImageTextSample)
            print(f"  [{s.meta.sample_id}] {s.text[:120]}")
        break
