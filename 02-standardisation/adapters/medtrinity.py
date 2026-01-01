import glob
import io
import logging
import os
import pickle
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import zstandard as zstd
from PIL import Image

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import BaseDataset, ImageSample, ImageTextSample, Sample, SampleMetadata

allowed_sources = [
    "deeplesion",
    "brats24",
    "brats",
    "pmc_oa",
    "breast_histo",
    "brats_24",
    "quilt_1m",
    "MAMA-MIA",
    "quilt_llava",
    "NCT-CRC-HE-100K",
    "adc22",
    "llava_med_data",
    "2u1-data5-nature",
    "4u8_data2_nature",
    "nih_chest",
    # "sammed",
    # "ct_rate",
    # "PTCGA",
    # "TCGA",
    # "bhx",
    # "cr_rate",
    # "VALSET",
    # "chexpert",
    # "pmc_vqa",
    # "uls23",
    # "ihc4bc",
    # "padchest",
    # "kipa22_kits23_cervix",
    # "CISC",
    # "flare23",
]


def _decode_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode in ("P", "RGBA", "LA"):
            img = img.convert("RGBA")
        return img.convert("RGB")
    except Exception:
        return None


class MedTrinityFullAdapter(BaseDataset):
    def __init__(
        self,
        data_dir: str,
        image_only: bool,
        cache_file: str,
        decode_workers: int,
    ):
        self.data_dir = data_dir
        self.decode_workers = decode_workers
        self.allowed_sources = set(allowed_sources)
        self.img_only = image_only

        self.parquet_dir = os.path.join(data_dir, "25M_full")
        self.tar_dir = os.path.join(data_dir, "25M_accessible")

        parquet_files = sorted(glob.glob(os.path.join(self.parquet_dir, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"no parquet files in {self.parquet_dir}")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.meta_lookup = pickle.load(f)
        else:
            dfs = []
            for pf in parquet_files:
                df = pd.read_parquet(pf, columns=["file_name", "caption", "source"])
                dfs.append(df)
            if not dfs:
                raise RuntimeError("Failed to load any metadata files.")
            full_df = pd.concat(dfs)

            self.meta_lookup = {}

            full_df = full_df[full_df["source"].isin(list(self.allowed_sources))]
            for _, row in full_df.iterrows():
                self.meta_lookup[row["file_name"]] = {
                    "caption": row["caption"],
                    "source": row["source"],
                }

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(self.meta_lookup, f)

    @property
    def id(self):
        return "medtrinity_full"

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        counter = 0

        # read tar.zst files
        tar_files = sorted(glob.glob(os.path.join(self.tar_dir, "*.tar.zst")))
        if not tar_files:
            logger.error(f"No .tar.zst files found in {self.tar_dir}")
            return

        logger.info(f"Found {len(tar_files)} tar shards.")
        if skip > 0:
            logger.info(f"Skipping first {skip} valid images.")

        dctx = zstd.ZstdDecompressor()
        pending = []  # (sample_id, image_bytes, caption, metadata_dict)

        with ThreadPoolExecutor(max_workers=self.decode_workers) as executor:
            for tar_path in tar_files:
                logger.info(f"Processing shard: {tar_path}")
                with (
                    open(tar_path, "rb") as ifh,
                    dctx.stream_reader(ifh) as reader,
                    tarfile.open(fileobj=reader, mode="r|") as tar,
                ):
                    for member in tar:
                        if not member.isfile():
                            continue

                        file_name = os.path.basename(member.name)
                        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                            logger.warning(f"skipping {file_name} (bad format)")
                            continue

                        meta = self.meta_lookup.get(file_name)
                        if not meta:
                            # missing or blacklisted source
                            continue

                        if counter < skip:
                            counter += 1
                            continue

                        # load image and add to pending
                        f_obj = tar.extractfile(member)
                        source = meta["source"]
                        if not f_obj:
                            continue
                        img_bytes = f_obj.read()
                        caption = meta["caption"]
                        meta_data = {
                            "dataset_id": self.id,
                            "source": source,
                            "file_name": file_name,
                        }

                        pending.append((counter, img_bytes, caption, meta_data))
                        counter += 1

                        if len(pending) >= batch_size:
                            yield self._decode_batch(executor, pending, logger)
                            pending = []

                        if counter % 10000 == 0:
                            logger.info(f"Streamed {counter} samples...")

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
                logger.warning(f"Failed to decode image {meta_data['file_name']}")
                continue
            if not self.img_only and text is None:
                logger.warning(f"Skipping {meta_data['file_name']}; missing caption")
                continue
            m = SampleMetadata(
                dataset_id=self.id,
                sample_id=sample_id,
                data=meta_data,
            )
            if self.img_only:
                batch.append(ImageSample(image=img, meta=m))
            else:
                batch.append(ImageTextSample(image=img, text=text, meta=m))
        return batch


# === FOR MANUAL TESTING CODE ===

if __name__ == "__main__":
    logger = logging.getLogger("medtrinity_full")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info("Initializing MedTrinityFullAdapter...")

    a = MedTrinityFullAdapter(
        data_dir="/capstor/store/cscs/swissai/infra01/medical/raw/medtrinity_25m",
        cache_file="/iopsstor/scratch/cscs/tchu/.cache/medtrinity/metadata_legal.pkl",
        image_only=True,
        decode_workers=100,
    )

    logger.info("Starting MedTrinityFullAdapter test stream...")
    for batch in a.stream(logger=logger, batch_size=1000):
        for b in batch:
            if a.img_only:
                assert isinstance(b, ImageSample)
            else:
                assert isinstance(b, ImageTextSample)
            print(
                "obtained sample:",
                b.meta.sample_id,
                b.image.size,  # type: ignore
                b.text[:20] if isinstance(b, ImageTextSample) else "",
                b.meta.data["file_name"],
            )
