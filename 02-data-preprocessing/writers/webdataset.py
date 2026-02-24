import io
import json
import logging
import os

import webdataset as wds

from pipeline.base import BaseWriter
from pipeline.schema import ImageSample, ImageTextSample, MultiImageTextSample, Sample

logger = logging.getLogger("pipeline.writers.webdataset")


def _serialize_image(image) -> tuple[bytes, str]:
    """Serialize a PIL image to bytes, returning (bytes, extension)."""
    buf = io.BytesIO()
    fmt = image.format or "PNG"

    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if fmt.upper() in ("JPEG", "JPG") and image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")
    if "transparency" in image.info:
        del image.info["transparency"]

    image.save(buf, format=fmt)
    ext = fmt.lower()
    if ext == "jpeg":
        ext = "jpg"
    return buf.getvalue(), ext


class WebDatasetWriter(BaseWriter):
    """Writes image-text pairs as webdataset tar shards.

    Each sample is stored as:
        __key__: "{dataset_id}__{sample_id:08d}"
        {ext}:   image bytes (png/jpg)
        txt:     caption text (utf-8)
    """

    def __init__(
        self,
        output_dir: str,
        maxcount: int = 10000,
    ):
        self.output_dir = output_dir
        self.maxcount = maxcount

        self._data_dir: str | None = None
        self._sink: wds.ShardWriter | None = None
        self._total_samples = 0
        self._dataset_samples = 0

    def open(self, dataset_id: str):
        self._data_dir = os.path.join(self.output_dir, dataset_id)
        os.makedirs(self._data_dir, exist_ok=True)
        self._dataset_samples = 0

        # count existing shards to resume
        existing = sorted(f for f in os.listdir(self._data_dir) if f.endswith(".tar"))
        start_shard = len(existing)

        stats_path = os.path.join(self._data_dir, ".stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
                self._total_samples = stats.get("total_samples", 0)
                self._dataset_samples = stats.get("dataset_samples", 0)

        pattern = os.path.join(self._data_dir, "part-%06d.tar")
        self._sink = wds.ShardWriter(
            pattern,
            maxcount=self.maxcount,
            start_shard=start_shard,
        )

        if start_shard > 0:
            logger.info(
                f"[{dataset_id}] Resuming from shard {start_shard} "
                f"({self._dataset_samples:,} samples)"
            )
        else:
            logger.info(f"[{dataset_id}] Starting fresh write")

    def write_batch(self, samples: list[Sample]):
        if not samples or self._sink is None:
            return

        for sample in samples:
            key = f"{sample.meta.dataset_id}__{sample.meta.sample_id:08d}"

            if isinstance(sample, MultiImageTextSample):
                record: dict = {"__key__": key}
                for i, image in enumerate(sample.images, start=1):
                    img_bytes, ext = _serialize_image(image)
                    record[f"img{i}.{ext}"] = img_bytes
                record["txt"] = sample.text.encode("utf-8")
                self._sink.write(record)

            elif isinstance(sample, ImageTextSample):
                img_bytes, ext = _serialize_image(sample.image)
                self._sink.write(
                    {
                        "__key__": key,
                        ext: img_bytes,
                        "txt": sample.text.encode("utf-8"),
                    }
                )

            elif isinstance(sample, ImageSample):
                img_bytes, ext = _serialize_image(sample.image)
                self._sink.write(
                    {
                        "__key__": key,
                        ext: img_bytes,
                        "txt": b"",
                    }
                )

            else:
                continue

            self._total_samples += 1
            self._dataset_samples += 1

    def close(self):
        if self._sink is not None:
            self._sink.close()
            self._sink = None

        if self._data_dir is not None:
            stats = {
                "total_samples": self._total_samples,
                "dataset_samples": self._dataset_samples,
            }
            with open(os.path.join(self._data_dir, ".stats.json"), "w") as f:
                json.dump(stats, f, indent=2)

        logger.info(f"Wrote {self._total_samples:,} total samples")
