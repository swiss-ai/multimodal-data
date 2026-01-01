import io
import json
import logging
import os

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


class HuggingFaceDatasetWriter(BaseWriter):
    def __init__(
        self,
        output_dir: str,
        target_shard_bytes: int,
    ):
        self.output_dir = output_dir
        self.data_dir = os.path.join(output_dir, "data")
        self.target_shard_bytes = target_shard_bytes

        self._writer: pq.ParquetWriter | None = None
        self._shard_idx = 0
        self._shard_bytes = 0
        self._shard_samples = 0
        self._total_samples = 0
        self._total_bytes = 0

    def open(self):
        os.makedirs(self.data_dir, exist_ok=True)

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

        for sample in samples:
            if not isinstance(sample, (ImageSample, ImageTextSample)):
                logger.warning(f"Skipping non-image sample: {type(sample)}")
                continue

            self._ensure_writer_open()
            row_bytes = self._write_sample(sample)
            self._shard_samples += 1
            self._shard_bytes += row_bytes
            self._total_samples += 1
            self._total_bytes += row_bytes

            if self._shard_bytes >= self.target_shard_bytes:
                self._close_shard()

        self._save_state()

    def close(self):
        self._close_shard()
        self._write_dataset_info()
        self._rename_shards()
        self._save_state(final=True)
        logger.info(f"Wrote {self._total_samples} samples in {self._shard_idx} shards")

    def _ensure_writer_open(self):
        if self._writer is None:
            shard_path = os.path.join(
                self.data_dir, f"train-{self._shard_idx:05d}.parquet"
            )
            self._writer = pq.ParquetWriter(shard_path, ARROW_SCHEMA)
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

    def _write_sample(self, sample: ImageSample | ImageTextSample) -> int:
        """Write a single sample, return bytes written."""
        assert self._writer is not None

        image_bytes = self._image_to_bytes(sample.image)

        table = pa.table(
            {
                "dataset_id": [sample.meta.dataset_id],
                "sample_id": [sample.meta.sample_id],
                "image": [{"bytes": image_bytes, "path": None}],
            },
            schema=ARROW_SCHEMA,
        )
        self._writer.write_table(table)

        return len(image_bytes)

    def _image_to_bytes(self, image) -> bytes:
        """Convert PIL image to bytes (preserve original format)."""
        buf = io.BytesIO()
        fmt = image.format or "PNG"

        if fmt.upper() in ("JPEG", "JPG") and image.mode == "RGBA":
            image = image.convert("RGB")

        image.save(buf, format=fmt)
        return buf.getvalue()

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
            "features": {
                "dataset_id": {"dtype": "string", "_type": "Value"},
                "sample_id": {"dtype": "int64", "_type": "Value"},
                "image": {"_type": "Image"},
            },
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
