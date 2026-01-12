import io
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
from datasets import Features, Image, Value

from pipeline.base import BaseWriter
from pipeline.schema import ImageSample, ImageTextSample, Sample

logger = logging.getLogger("pipeline.writers.huggingface")

ARROW_MAX_BYTES = 1_500_000_000
HF_FEATURES = Features(
    {
        "dataset_id": Value("string"),
        "sample_id": Value("int64"),
        "image": Image(),
    }
)
SCHEMA = HF_FEATURES.arrow_schema


def _serialize_sample(sample: ImageSample | ImageTextSample) -> tuple[str, int, bytes]:
    image = sample.image
    buf = io.BytesIO()
    fmt = image.format or "PNG"

    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if fmt.upper() in ("JPEG", "JPG") and image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")
    if "transparency" in image.info:
        del image.info["transparency"]

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
        self.target_shard_bytes = target_shard_bytes
        self.num_workers = num_workers

        self._data_dir: str | None = None
        self._sink: pa.OSFile | None = None
        self._writer: pa.RecordBatchStreamWriter | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._shard_idx = 0
        self._shard_bytes = 0
        self._shard_samples = 0
        self._total_bytes = 0
        self._total_samples = 0

    def open(self, dataset_id: str):
        self._data_dir = os.path.join(self.output_dir, dataset_id)
        os.makedirs(self._data_dir, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=self.num_workers)

        existing = sorted(f for f in os.listdir(self._data_dir) if f.endswith(".arrow"))
        self._shard_idx = len(existing)

        stats_path = os.path.join(self._data_dir, ".stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
                self._total_bytes = stats.get("total_bytes", 0)
                self._total_samples = stats.get("total_samples", 0)
        else:
            self._total_bytes = 0
            self._total_samples = 0

        if self._shard_idx > 0:
            logger.info(
                f"[{dataset_id}] Resuming from shard {self._shard_idx} "
                f"({self._total_samples:,} samples, {self._total_bytes:,} bytes)"
            )
        else:
            logger.info(f"[{dataset_id}] Starting fresh write")

    def write_batch(self, samples: list[Sample]):
        if not samples:
            return

        image_samples = [
            s for s in samples if isinstance(s, (ImageSample, ImageTextSample))
        ]
        if not image_samples:
            return

        assert self._executor is not None
        results = list(self._executor.map(_serialize_sample, image_samples))

        batch_bytes = sum(len(img_bytes) for _, _, img_bytes in results)
        if batch_bytes < ARROW_MAX_BYTES:
            self._write_results(results)
            return

        chunk, chunk_bytes = [], 0
        for r in results:
            nbytes = len(r[2])
            if chunk_bytes + nbytes > ARROW_MAX_BYTES and chunk:
                self._write_results(chunk)
                chunk = []
                chunk_bytes = 0
            chunk.append(r)
            chunk_bytes += nbytes
        if chunk:
            self._write_results(chunk)

    def _write_results(self, results: list[tuple[str, int, bytes]]):
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
            schema=SCHEMA,
        )
        assert self._writer is not None
        self._writer.write_table(table)

        self._shard_samples += len(results)
        self._shard_bytes += batch_bytes
        self._total_samples += len(results)
        self._total_bytes += batch_bytes
        if self._shard_bytes >= self.target_shard_bytes:
            self._close_shard()

    def _ensure_writer_open(self):
        if self._writer is None:
            assert self._data_dir is not None
            shard_path = os.path.join(
                self._data_dir, f"data-{self._shard_idx:05d}.arrow"
            )
            self._sink = pa.OSFile(shard_path, "wb")
            self._writer = pa.ipc.new_stream(self._sink, SCHEMA)

    def _close_shard(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._sink is not None:
            self._sink.close()
            self._sink = None
            self._shard_idx += 1
            self._shard_bytes = 0
            self._shard_samples = 0

    def close(self):
        self._close_shard()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._finalize_hf_metadata()
        logger.info(
            f"Wrote {self._shard_idx} shards, "
            f"{self._total_samples:,} samples, {self._total_bytes:,} bytes"
        )

    def _finalize_hf_metadata(self):
        assert self._data_dir is not None

        # .stats.json - for fast resume
        stats = {
            "total_bytes": self._total_bytes,
            "total_samples": self._total_samples,
        }
        with open(os.path.join(self._data_dir, ".stats.json"), "w") as f:
            json.dump(stats, f)

        # state.json - for load_from_disk
        files = sorted(f for f in os.listdir(self._data_dir) if f.endswith(".arrow"))
        state = {
            "_data_files": [{"filename": f} for f in files],
            "_fingerprint": "pipeline_build",
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": "train",
        }
        with open(os.path.join(self._data_dir, "state.json"), "w") as f:
            json.dump(state, f, indent=2)

        # dataset_info.json
        info_dict = {
            "features": HF_FEATURES.to_dict(),
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": self._total_bytes,
                    "num_examples": self._total_samples,
                }
            },
        }
        with open(os.path.join(self._data_dir, "dataset_info.json"), "w") as f:
            json.dump(info_dict, f, indent=2)
