import io
import json
import logging
import os
import tarfile

from pipeline.base import BaseSink
from pipeline.schema import ImageSample, ImageTextSample, Sample, TextSample

logger = logging.getLogger("pipeline.sinks.webdataset")


class WebDatasetSink(BaseSink):
    """WebDataset writer sink."""

    def __init__(
        self,
        output_dir: str,
        samples_per_shard: int = 10000,
        target_shard_bytes: int = 500_000_000,
        image_format: str = "jpeg",
    ):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.target_shard_bytes = target_shard_bytes
        self.image_format = image_format.lower()

        self._tar: tarfile.TarFile | None = None
        self._shard_idx = 0
        self._shard_sample_count = 0
        self._shard_bytes = 0
        self._total_samples = 0

    def open(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Writing to {self.output_dir}")

    def write_batch(self, samples: list[Sample]) -> None:
        for sample in samples:
            self._ensure_shard_open()
            self._write_sample(sample)
            self._shard_sample_count += 1
            self._total_samples += 1

            if (
                self._shard_sample_count >= self.samples_per_shard
                or self._shard_bytes >= self.target_shard_bytes
            ):
                self._close_shard()

    def close(self) -> None:
        self._close_shard()
        logger.info(f"Wrote {self._total_samples} samples in {self._shard_idx} shards")

    def _ensure_shard_open(self) -> None:
        if self._tar is None:
            shard_path = os.path.join(self.output_dir, f"{self._shard_idx:06d}.tar")
            self._tar = tarfile.open(shard_path, "w")
            logger.debug(f"Opened shard {shard_path}")

    def _close_shard(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
            mb = self._shard_bytes / 1_000_000
            logger.debug(
                f"Closed shard {self._shard_idx}"
                f" ({self._shard_sample_count} samples,"
                f" {mb:.2f}MB)"
            )
            self._shard_idx += 1
            self._shard_sample_count = 0
            self._shard_bytes = 0

    def _write_sample(self, sample: Sample) -> None:
        assert self._tar is not None
        key = f"{sample.meta.sample_id:09d}"

        meta_json = json.dumps(sample.meta.data).encode("utf-8")
        self._add_bytes(f"{key}.json", meta_json)

        if isinstance(sample, TextSample):
            self._add_bytes(f"{key}.txt", sample.text.encode("utf-8"))

        elif isinstance(sample, ImageSample):
            img_bytes = self._image_to_bytes(sample.image)
            self._add_bytes(f"{key}.{self.image_format}", img_bytes)

        elif isinstance(sample, ImageTextSample):
            img_bytes = self._image_to_bytes(sample.image)
            self._add_bytes(f"{key}.{self.image_format}", img_bytes)
            self._add_bytes(f"{key}.txt", sample.text.encode("utf-8"))

    def _image_to_bytes(self, image) -> bytes:
        buf = io.BytesIO()
        fmt = self.image_format.upper()
        if fmt == "JPEG" and image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(buf, format=fmt)
        return buf.getvalue()

    def _add_bytes(self, name: str, data: bytes) -> None:
        assert self._tar is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        self._tar.addfile(info, io.BytesIO(data))
        self._shard_bytes += len(data)
