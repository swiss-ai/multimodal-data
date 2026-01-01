import io
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from datasets import load_from_disk
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, ImageTextSample, Sample, SampleMetadata


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


class MeditronImageAdapter(BaseDataset):
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
        counter = 0

        logger.info(f"Starting stream for {self.id} from {self.data_dir}")
        if skip > 0:
            logger.info(f"Skipping first {skip} images.")

        pending = []  # (sample_id, texts, image_bytes)

        self.dataset: list[dict]
        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
            for sample in self.dataset:
                texts = sample["conversations"]
                for item in sample["modalities"]:
                    if item["type"] != "image":
                        continue

                    if counter < skip:
                        counter += 1
                        continue

                    image_bytes = item["value"]["bytes"]
                    pending.append((counter, image_bytes, texts))
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
        pending: list[tuple[int, bytes, list[dict]]],
        logger,
    ) -> list[Sample]:
        """Decode a batch of images in parallel."""
        sample_ids = [p[0] for p in pending]
        image_bytes_list = [p[1] for p in pending]
        texts_list = [p[2] for p in pending]

        pil_images = list(executor.map(_decode_image, image_bytes_list))

        batch = []
        for sample_id, img, texts in zip(sample_ids, pil_images, texts_list):
            if img is None:
                logger.warning(f"Failed to decode image for sample {sample_id}")
                continue
            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=sample_id,
                data={"dataset_id": self.id},
            )
            if self.image_only:
                batch.append(ImageSample(image=img, meta=m))
            else:
                text = "\n".join([f"{t['role']}: {t['content']}" for t in texts])
                batch.append(ImageTextSample(image=img, text=text, meta=m))
        return batch


if __name__ == "__main__":
    logger = logging.getLogger("meditron_full")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    datasets = [
        {
            "dataset_id": "busi",
            "data_dir": "/capstor/store/cscs/swissai/infra01/medical/meditron/BUSI",
        },
        {
            "dataset_id": "covid_us",
            "data_dir": "/capstor/store/cscs/swissai/infra01/medical/meditron/COVID_US",
        },
        {
            "dataset_id": "ddti",
            "data_dir": "/capstor/store/cscs/swissai/infra01/medical/meditron/DDTI",
        },
        {
            "dataset_id": "llava_instruct",
            "data_dir": "/capstor/store/cscs/swissai/infra01/medical/meditron/llava_instruct",
        },
        {
            "dataset_id": "llava_pretrain_cleaned",
            "data_dir": "/capstor/store/cscs/swissai/infra01/medical/meditron/llava_pretrain_cleaned",
        },
        {
            "dataset_id": "mammoth",
            "data_dir": "/capstor/store/cscs/swissai/infra01/medical/meditron/image_mammoth",
        },
        {
            "dataset_id": "pixmo_anything",
            "data_dir": "/capstor/store/cscs/swissai/infra01/medical/meditron/pixmo_anything",
        },
        {
            "dataset_id": "pixmo_cap",
            "data_dir": "/capstor/store/cscs/swissai/infra01/medical/meditron/pixmo_cap",
        },
    ]
    for ds in datasets:
        print(f"Testing dataset: {ds['dataset_id']}")

        a = MeditronImageAdapter(
            dataset_id=ds["dataset_id"],
            data_dir=ds["data_dir"],
            image_only=True,
            decode_workers=100,
        )

        # print(a.dataset)
        # sample = a.dataset[0]
        # mods = sample["modalities"]
        # texts = sample["conversations"]
        # for mod in mods:
        #     if mod["type"] != "image":
        #         continue
        #     img_bytes = mod["value"]["bytes"]
        for batch in a.stream(logger=logger, skip=0, batch_size=1000):
            for b in batch:
                print(
                    "obtained sample:",
                    b.meta.sample_id,
                    b.image.size,  # type: ignore
                    b.text[:50] if isinstance(b, ImageTextSample) else None,
                )
