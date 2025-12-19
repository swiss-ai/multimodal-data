from logging import Logger
from typing import List

from src.adapter import BaseAdapter
from src.allowlist import AllowlistDB
from src.filter import BaseFilter
from src.schema import RawSample
from src.writer import ShardWriter

# TODO: batch processing via multiprocessing


class Pipeline:
    def __init__(
        self,
        logger: Logger,
        adapters: List[BaseAdapter],
        filters: List[BaseFilter],
    ):
        self.logger = logger
        self.adapters = adapters
        self.filters = filters

    def scan(self, allowlist_path: str, batch_size: int):
        """
        Iterates adapters, applies filters, and populates the Allowlist.
        """
        with AllowlistDB(allowlist_path) as allowlist:
            for adapter in self.adapters:
                self.logger.info(f"scanning adapter: {adapter.name}")

                batch_buffer = []

                for sample in adapter.stream():
                    self.logger.debug(f"scanning sample: {sample.meta.sample_id}")

                    if not self.apply_filters(sample):
                        continue

                    batch_buffer.append((sample.meta.dataset_id, sample.meta.sample_id))

                    if len(batch_buffer) >= batch_size:
                        allowlist.add_batch(batch_buffer)
                        batch_buffer = []

                # flush remaining
                if batch_buffer:
                    allowlist.add_batch(batch_buffer)

                self.logger.info(f"completed scanning adapter: {adapter.name}")

    def build(self, allowlist_path: str, output_dir: str):
        """
        Iterates adapters (again), checks Allowlist, hydrates, and writes WebDatasets.
        """
        with AllowlistDB(allowlist_path) as allowlist:
            for adapter in self.adapters:
                self.logger.info(f"building adapter: {adapter.name}")

                shard_name = f"{adapter.name}_part"

                with ShardWriter(self.logger, output_dir, shard_name) as writer:
                    for sample in adapter.stream():
                        self.logger.debug(f"building sample: {sample.meta.sample_id}")

                        if allowlist.exists(adapter.name, sample.meta.sample_id):
                            writer.write(sample)

                self.logger.info(f"completed building adapter: {adapter.name}")

    def apply_filters(self, sample: RawSample) -> bool:
        for filter in self.filters:
            if not filter(sample):
                return False
        return True
