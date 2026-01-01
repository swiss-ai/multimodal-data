import glob
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

from pipeline import BaseDataset, ImageSample, ImageTextSample, Sample, SampleMetadata


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


class MedMaxImageAdapter(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        image_only: bool,
        cache_file: str,
        decode_workers: int,
    ):
        self.data_dir = data_dir
        self.decode_workers = decode_workers
        self.image_only = image_only

        search_pattern = os.path.join(data_dir, "*tar.gz")
        self.chunk_files = sorted(glob.glob(search_pattern))
        if not self.chunk_files:
            raise FileNotFoundError(f"No tar chunks found in {data_dir}.'")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.meta_lookup = pickle.load(f)
        else:
            train_jsonl = os.path.join(data_dir, "train.jsonl")
            valid_jsonl = os.path.join(data_dir, "validation.jsonl")
            if not os.path.exists(train_jsonl) or not os.path.exists(valid_jsonl):
                raise FileNotFoundError(
                    f"no {{train,valididation}}.jsonl in {data_dir}."
                )

            self.meta_lookup = {}

            for jsonl_file in [valid_jsonl, train_jsonl]:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        item = json.loads(line)
                        self.meta_lookup[item["image_path"]] = {
                            "text": item.get("text", ""),
                            "task": item.get("task", ""),
                            "source": item.get("source", ""),
                        }

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.meta_lookup, f)

    @property
    def id(self):
        return "medmax"

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0
        if skip > 0:
            logger.info(f"Skipping first {skip} images.")

        pending = []  # (sample_id, image_bytes, text)

        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
            for tar_path in self.chunk_files:
                logger.info(f"Streaming from tar chunk: {tar_path}")
                with tarfile.open(tar_path, mode="r|gz") as tar:
                    for member in tar:
                        if not member.isfile():
                            continue

                        file_path = member.name
                        if not member.name.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
                        ):
                            logger.warning(f"skipping {file_path} (bad format)")
                            continue

                        meta = self.meta_lookup.get(file_path)
                        if not meta:
                            # missing or blacklisted source
                            continue

                        f_obj = tar.extractfile(member)
                        if not f_obj:
                            continue
                        try:
                            img_bytes = f_obj.read()
                        except Exception:
                            logger.warning(f"failed to read {file_path}")
                            continue

                        if counter < skip:
                            counter += 1
                            continue

                        text = meta["text"]
                        task = meta["task"]
                        source = meta["source"]
                        meta = SampleMetadata(
                            dataset_id=self.id,
                            sample_id=counter,
                            data={
                                "file_path": file_path,
                                "task": task,
                                "source": source,
                            },
                        )

                        pending.append((img_bytes, text, meta))
                        counter += 1

                        if len(pending) >= batch_size:
                            yield self._decode_batch(executor, pending)
                            pending = []

                        if counter % 10000 == 0:
                            logger.debug(f"Streamed {counter} images so far.")

            if pending:
                yield self._decode_batch(executor, pending)
        logger.info("Finished streaming.")

    def _decode_batch(
        self,
        executor: ThreadPoolExecutor,
        pending: list[tuple[bytes, str, SampleMetadata]],
    ) -> list[Sample]:
        image_bytes_list = [p[0] for p in pending]
        texts = [p[1] for p in pending]
        metas = [p[2] for p in pending]

        pil_images = list(executor.map(_decode_image, image_bytes_list))

        batch = []
        for img, text, meta in zip(pil_images, texts, metas):
            if img is None:
                logging.warning(f"failed to decode {meta.data['file_path']}")
                continue
            if self.image_only:
                batch.append(ImageSample(image=img, meta=meta))
            else:
                batch.append(ImageTextSample(image=img, text=text, meta=meta))
        return batch


if __name__ == "__main__":
    logger = logging.getLogger("medmax")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info("Initializing medmax...")

    a = MedMaxImageAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/medmax_data",
        cache_file="/iopsstor/scratch/cscs/tchu/.cache/medmax/metadata.pkl",
        image_only=True,
        decode_workers=100,
    )

    logger.info("Starting medmax test stream...")
    print("Dataset ID:", a.id)
    print(len(a.meta_lookup))
    for batch in a.stream(logger=logger, batch_size=1000):
        for b in batch:
            print(
                "obtained sample:",
                b.meta.sample_id,
                b.image.size,  # type: ignore
                b.meta.data["file_path"],
            )
