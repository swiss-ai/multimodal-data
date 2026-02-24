import io
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from datasets import load_from_disk
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    BaseDataset,
    ImageSample,
    MultiImageTextSample,
    Sample,
    SampleMetadata,
)

# Placeholder token used in the source datasets
SRC_IMAGE_TOKEN = "<|reserved_special_token_0|>"
# Output token pattern: <|img1|>, <|img2|>, ...
IMG_TOKEN = "<|img{n}|>"


def _make_img_tokens(n: int) -> list[str]:
    return [IMG_TOKEN.format(n=i + 1) for i in range(n)]


def _replace_image_tokens(conversations: list[dict], n_images: int) -> str:
    """
    Join conversation turns and replace each occurrence of the source image
    placeholder with <|img1|>, <|img2|>, ... in order.

    If the text has fewer placeholders than images, the missing tokens are
    prepended. If it has more, the extras are left as-is (shouldn't happen).
    """
    text = "\n".join(f"{t['role']}: {t['content']}" for t in conversations)
    img_tokens = _make_img_tokens(n_images)

    token_iter = iter(img_tokens)
    def _replacer(_match):
        try:
            return next(token_iter)
        except StopIteration:
            return _match.group(0)  # leave extra tokens untouched

    text = re.sub(re.escape(SRC_IMAGE_TOKEN), _replacer, text)

    # Prepend any tokens that weren't placed in the text
    remaining = list(token_iter)
    if remaining:
        text = "".join(remaining) + "\n" + text

    return text


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


def _decode_images(images_bytes: list[bytes]) -> list[Image.Image | None]:
    return [_decode_image(b) for b in images_bytes]


class MeditronAdapter(BaseDataset):
    def __init__(
        self,
        dataset_id: str,
        data_dir: str,
        image_only: bool,
        decode_workers: int,
    ):
        self.data_dir = data_dir
        self._id = dataset_id
        self.decode_workers = decode_workers
        self.dataset = load_from_disk(data_dir)  # type: ignore
        self.image_only = image_only

    @property
    def id(self):
        return self._id

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0  # counts dataset samples (groups of images), not individual images

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")
        if skip > 0:
            logger.info(f"Skipping first {skip} samples.")

        # pending: list of (sample_id, images_bytes_list, conversations)
        pending = []

        self.dataset: list[dict]
        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
            for sample in self.dataset:
                if counter < skip:
                    counter += 1
                    continue

                conversations = sample["conversations"]
                images_bytes = [
                    item["value"]["bytes"]
                    for item in sample["modalities"]
                    if item["type"] == "image"
                ]

                if not images_bytes:
                    counter += 1
                    continue

                pending.append((counter, images_bytes, conversations))
                counter += 1

                if len(pending) >= batch_size:
                    yield self._decode_batch(executor, pending, logger)
                    pending = []

                if counter % 10000 == 0:
                    logger.info(f"Streamed {counter} samples so far...")

            if pending:
                yield self._decode_batch(executor, pending, logger)

        logger.info("Finished streaming.")

    def _decode_batch(
        self,
        executor: ThreadPoolExecutor,
        pending: list[tuple[int, list[bytes], list[dict]]],
        logger,
    ) -> list[Sample]:
        """Decode a batch of samples (each may have multiple images)."""
        batch = []
        futures = [
            executor.submit(_decode_images, images_bytes)
            for _, images_bytes, _ in pending
        ]
        for (sample_id, _, conversations), future in zip(pending, futures):
            pil_images = future.result()

            # Drop any images that failed to decode
            valid = [img for img in pil_images if img is not None]
            n_failed = len(pil_images) - len(valid)
            if n_failed:
                logger.warning(
                    f"Sample {sample_id}: {n_failed}/{len(pil_images)} images failed to decode"
                )
            if not valid:
                continue

            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=sample_id,
                data={"dataset_id": self.id},
            )

            if self.image_only:
                if len(valid) == 1:
                    batch.append(ImageSample(image=valid[0], meta=m))
                else:
                    # For image-only multi-image, emit one ImageSample per image
                    for img in valid:
                        batch.append(ImageSample(image=img, meta=m))
            else:
                text = _replace_image_tokens(conversations, len(valid))
                batch.append(MultiImageTextSample(images=valid, text=text, meta=m))

        return batch


if __name__ == "__main__":
    logger = logging.getLogger("meditron_full")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    BASE = "/capstor/store/cscs/swissai/infra01/vision-datasets/medical/raw/meditron"
    datasets = [
        {"dataset_id": "busi", "data_dir": f"{BASE}/BUSI"},
        {"dataset_id": "iu_xray", "data_dir": f"{BASE}/iu_xray"},
    ]
    for ds in datasets:
        print(f"\n=== Testing dataset: {ds['dataset_id']} ===")
        a = MeditronAdapter(
            dataset_id=ds["dataset_id"],
            data_dir=ds["data_dir"],
            image_only=False,
            decode_workers=4,
        )
        for batch in a.stream(logger=logger, skip=0, batch_size=4):
            for b in batch:
                if isinstance(b, MultiImageTextSample):
                    print(f"  MultiImage ({len(b.images)} imgs): {b.text[:100]}")
                elif isinstance(b, ImageTextSample):
                    print(f"  Single: {b.text[:100]}")
            break
