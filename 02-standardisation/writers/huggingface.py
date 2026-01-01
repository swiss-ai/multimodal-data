import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.base import BaseWriter
from pipeline.schema import ImageSample, ImageTextSample, Sample

logger = logging.getLogger("pipeline.writers.huggingface")

ARROW_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("sample_id", pa.int64()),
        (
            "image",
            pa.struct(
                [
                    ("bytes", pa.binary()),
                    ("path", pa.string()),
                ]
            ),
        ),
    ]
)

HF_METADATA = {
    "info": {
        "features": {
            "dataset_id": {"dtype": "string", "_type": "Value"},
            "sample_id": {"dtype": "int64", "_type": "Value"},
            "image": {"_type": "Image"},
        }
    }
}


def _serialize_sample(sample: ImageSample | ImageTextSample) -> tuple[str, int, bytes]:
    image = sample.image
    buf = io.BytesIO()
    fmt = image.format or "PNG"

    if fmt.upper() in ("JPEG", "JPG") and image.mode == "RGBA":
        image = image.convert("RGB")

    image.save(buf, format=fmt)
    return sample.meta.dataset_id, sample.meta.sample_id, buf.getvalue()


class HuggingFaceDatasetWriter(BaseWriter):
    def __init__(
        self,
        output_dir: str,
        target_shard_bytes: int,
        num_workers: int,
    ):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        self.target_shard_bytes = target_shard_bytes
        self.num_workers = num_workers

        metadata = ARROW_SCHEMA.metadata or {}
        metadata[b"huggingface"] = json.dumps(HF_METADATA).encode("utf-8")
        self.schema = ARROW_SCHEMA.with_metadata(metadata)

        self._writer: pq.ParquetWriter | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._shard_idx = 0
        self._shard_bytes = 0
        self._shard_samples = 0
        self._total_samples = 0
        self._total_bytes = 0

    def open(self):
        os.makedirs(self.data_dir, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)

        state_path = os.path.join(self.output_dir, "state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)
            self._shard_idx = state["next_shard_idx"]
            self._total_samples = state["total_samples"]
            self._total_bytes = state["total_bytes"]
            logger.info(
                f"Resuming from shard {self._shard_idx} "
                f"({self._total_samples} samples written)"
            )
        else:
            logger.info(f"Starting fresh write to {self.output_dir}")

    def write_batch(self, samples: list[Sample]):
        if not samples:
            return

        image_samples = [
            s for s in samples if isinstance(s, (ImageSample, ImageTextSample))
        ]
        if not image_samples:
            return

        # serialize images
        assert self._executor is not None
        results = list(self._executor.map(_serialize_sample, image_samples))

        dataset_ids = []
        sample_ids = []
        images = []
        batch_bytes = 0

        for dataset_id, sample_id, img_bytes in results:
            dataset_ids.append(dataset_id)
            sample_ids.append(sample_id)
            images.append({"bytes": img_bytes, "path": None})
            batch_bytes += len(img_bytes)

        self._ensure_writer_open()
        table = pa.table(
            {
                "dataset_id": dataset_ids,
                "sample_id": sample_ids,
                "image": images,
            },
            schema=self.schema,
        )
        assert self._writer is not None
        self._writer.write_table(table)

        # update counters
        self._shard_samples += len(results)
        self._shard_bytes += batch_bytes
        self._total_samples += len(results)
        self._total_bytes += batch_bytes

        # rotate shard (if needed)
        if self._shard_bytes >= self.target_shard_bytes:
            self._close_shard()

        self._save_state()

    def close(self):
        self._close_shard()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._write_dataset_info()
        self._rename_shards()
        self._save_state(final=True)
        logger.info(f"Wrote {self._total_samples} samples in {self._shard_idx} shards")

    def _ensure_writer_open(self):
        if self._writer is None:
            shard_path = os.path.join(
                self.data_dir, f"train-{self._shard_idx:05d}.parquet"
            )
            self._writer = pq.ParquetWriter(shard_path, self.schema)
            logger.debug(f"Opened shard {shard_path}")

    def _close_shard(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            logger.debug(
                f"Closed shard {self._shard_idx} "
                f"({self._shard_samples} samples, {self._shard_bytes / 1e9:.2f}GB)"
            )
            self._shard_idx += 1
            self._shard_bytes = 0
            self._shard_samples = 0

    def _save_state(self, final: bool = False):
        state = {
            "next_shard_idx": self._shard_idx,
            "total_samples": self._total_samples,
            "total_bytes": self._total_bytes,
            "complete": final,
        }
        state_path = os.path.join(self.output_dir, "state.json")
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _write_dataset_info(self):
        info = {
            "builder_name": "parquet",
            "dataset_name": os.path.basename(self.output_dir),
            "features": HF_METADATA["info"]["features"],
            "splits": {
                "train": {
                    "name": "train",
                    "num_examples": self._total_samples,
                    "num_bytes": self._total_bytes,
                    "dataset_name": os.path.basename(self.output_dir),
                }
            },
        }
        info_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def _rename_shards(self):
        total_shards = self._shard_idx
        if total_shards == 0:
            return

        for i in range(total_shards):
            old_name = os.path.join(self.data_dir, f"train-{i:05d}.parquet")
            new_name = os.path.join(
                self.data_dir, f"train-{i:05d}-of-{total_shards:05d}.parquet"
            )
            if os.path.exists(old_name):
                os.rename(old_name, new_name)
