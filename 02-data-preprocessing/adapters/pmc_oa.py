import io
import logging
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor

import jsonlines
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageTextSample, Sample, SampleMetadata


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


class PMCOAAdapter(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        decode_workers: int,
    ):
        self.data_dir = data_dir
        self.decode_workers = decode_workers

        self.zip_path = os.path.join(data_dir, "images.zip")
        self.jsonl_path = os.path.join(data_dir, "pmc_oa.jsonl")
        self.img_subdir = "caption_T060_filtered_top4_sep_v0_subfigures"

        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"image not found at {self.zip_path}")
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"metadata jsonl not found at {self.jsonl_path}")

    @property
    def id(self):
        return "pmc_oa"

    def _load_metadata_lookup(self, logger):
        logger.info(f"Loading metadata from {self.jsonl_path} into memory...")
        lookup = {}
        with jsonlines.open(self.jsonl_path) as reader:
            for obj in reader:
                zip_member_name = f"{self.img_subdir}/{obj['image']}"
                lookup[zip_member_name] = {
                    "caption": obj["caption"],
                    "file_name": obj["image"],
                }
        logger.info(f"Loaded {len(lookup)} metadata entries.")
        return lookup

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0

        metadata_lookup = self._load_metadata_lookup(logger)
        pending = []  # (sample_id, image_bytes, caption, meta_dict)

        logger.info(f"Opening ZIP resource at {self.zip_path}...")
        with (
            zipfile.ZipFile(self.zip_path, "r") as zf,
            ThreadPoolExecutor(max_workers=self.decode_workers) as executor,
        ):
            for sample_id, zip_info in enumerate(zf.infolist()):
                if zip_info.filename not in metadata_lookup:
                    continue

                meta = metadata_lookup[zip_info.filename]
                if sample_id < skip:
                    continue

                img_bytes = zf.read(zip_info)
                metadata_out = {
                    "dataset_id": self.id,
                    "file_name": meta["file_name"],
                }

                pending.append((sample_id, img_bytes, meta["caption"], metadata_out))
                if len(pending) >= batch_size:
                    yield self._decode_batch(executor, pending, logger)
                    pending = []

                counter += 1
                if counter % 5000 == 0:
                    logger.info(f"streamed {counter} samples...")

            if pending:
                yield self._decode_batch(executor, pending, logger)

        logger.info("Finished streaming.")

    def _decode_batch(
        self,
        executor: ThreadPoolExecutor,
        pending: list[tuple[int, bytes, str, dict]],
        logger,
    ) -> list[Sample]:
        sample_ids = [p[0] for p in pending]
        image_bytes_list = [p[1] for p in pending]
        captions = [p[2] for p in pending]
        metas = [p[3] for p in pending]

        pil_images = list(executor.map(_decode_image, image_bytes_list))

        batch = []
        for sample_id, img, text, meta_data in zip(
            sample_ids, pil_images, captions, metas
        ):
            if img is None:
                logger.warning(f"failed to decode {meta_data['file_name']}")
                continue
            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=sample_id,
                data=meta_data,
            )
            batch.append(ImageTextSample(image=img, text=text, meta=m))
        return batch


if __name__ == "__main__":
    logger = logging.getLogger("pmc_oa")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Initializing...")

    a = PMCOAAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/pmc_oa",
        decode_workers=64,
    )

    logger.info("Starting test stream...")
    bi = 0
    for batch in a.stream(logger=logger, batch_size=1000):
        bi += 1
        logger.info(f"Processed batch {bi} with {len(batch)} samples.")
