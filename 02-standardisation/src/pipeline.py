from logging import Logger
from typing import List

from src.allowlist import AllowlistDB
from src.base import BaseDataset, BaseFilter
from src.schema import RawSample
from src.writer import ShardWriter


class Pipeline:
    def __init__(
        self,
        logger: Logger,
        datasets: List[BaseDataset],
        filters: List[BaseFilter],
        allowlist_path: str,
        batch_size: int,
        output_dir: str,
    ):
        self.logger = logger
        self.datasets = datasets
        self.filters = filters

        self.allowlist = AllowlistDB(allowlist_path)
        self.batch_size = batch_size

        self.output_dir = output_dir

    def scan(self):
        """
        Iterates datasets, applies filters, and populates the Allowlist.
        """
        batch_buffer = []

        for dataset in self.datasets:
            self.logger.info(f"scanning dataset adapter: {dataset.id}")

            for sample in dataset:
                self.logger.debug(f"scanning: {dataset.id}/{sample.meta.sample_id}")

                if not self._apply_filters(sample):
                    continue

                batch_buffer.append((sample.meta.dataset_id, sample.meta.sample_id))

                if len(batch_buffer) == self.batch_size:
                    self.allowlist.add_batch(batch_buffer)
                    batch_buffer = []

            self.logger.info(f"completed scanning dataset: {dataset.id}")

        # flush remaining
        if batch_buffer:
            self.allowlist.add_batch(batch_buffer)

    def build(self):
        """
        Iterates datasets for samples in the allowlist, and writes WebDatasets.
        """
        for dataset in self.datasets:
            self.logger.info(f"building dataset: {dataset.id}")

            shard_id = f"shard_{dataset.id}"

            with ShardWriter(self.logger, self.output_dir, shard_id) as writer:
                for sample in dataset:
                    self.logger.debug(f"building sample: {sample.meta.sample_id}")

                    if self.allowlist.exists(dataset.id, sample.meta.sample_id):
                        writer.write(sample)

            self.logger.info(f"completed building datasets: {dataset.id}")

    def _apply_filters(self, sample: RawSample) -> bool:
        for filter in self.filters:
            if not filter(sample):
                return False
        return True
