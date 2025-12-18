import io
import logging
import tarfile
import time
from pathlib import Path

import msgspec

from src.schema.sample import RawSample

# TODO:
# - split large shards
# - split by sample type


class ShardWriter:
    """
    Writes samples to a WebDataset tar archive.
    """

    def __init__(self, logger: logging.Logger, output_dir: str, shard_name: str):
        self.logger = logger

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tar_path = self.output_dir / f"{shard_name}.tar"
        self.tar = tarfile.open(self.tar_path, "w")

        self.json_encoder = msgspec.json.Encoder()

    def write(self, sample: RawSample):
        safe_id = sample.meta.sample_id.replace("/", "_")
        key = f"{sample.meta.dataset_name}/{safe_id}"

        self.logger.debug(f"writing sample: {key}")

        meta_bytes = self.json_encoder.encode(sample.meta)
        self.add_file(f"{key}.json", meta_bytes)

        files_map = sample.export_content()
        for extension, data in files_map.items():
            self.add_file(f"{key}.{extension}", data)

    def add_file(self, name: str, data: bytes):
        ti = tarfile.TarInfo(name)
        ti.size = len(data)
        ti.mtime = time.time()
        self.tar.addfile(ti, io.BytesIO(data))

    def close(self):
        self.tar.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
