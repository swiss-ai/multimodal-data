import logging
from collections.abc import Sequence

from pipeline.allowlist import Allowlist
from pipeline.base import BaseDataset, BaseWriter
from pipeline.checkpoint import Checkpoint
from pipeline.schema import Sample
from pipeline.workers import FilterFactory, WorkerPool

logger = logging.getLogger("pipeline")


class Pipeline:
    """Scans datasets, applies filters, builds manifest."""

    def __init__(
        self,
        datasets: Sequence[BaseDataset],
        filter_factories: Sequence[FilterFactory],
        writer: BaseWriter | None,
        manifest_db_path: str,
        checkpoint_db_path: str,
        num_workers: int,
        batch_size: int,
    ):
        self.datasets = datasets
        self.filter_factories = filter_factories
        self.writer = writer
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.allowlist = Allowlist(manifest_db_path)
        self.checkpoint = Checkpoint(checkpoint_db_path)

    def scan(self):
        logger.info(f"Starting scan of {len(self.datasets)} dataset(s)")

        if self.writer:
            self.writer.open()

        try:
            with WorkerPool(self.filter_factories, self.num_workers) as pool:
                for dataset in self.datasets:
                    self._scan_dataset(dataset, pool)
        finally:
            if self.writer:
                self.writer.close()

        logger.info("Scan complete")

    def _scan_dataset(self, dataset: BaseDataset, pool: WorkerPool):
        dataset_id = dataset.id

        if self.checkpoint.is_complete(dataset_id):
            logger.info(f"[{dataset_id}] Skipping (already complete)")
            return

        last_id = self.checkpoint.get_last_sample_id(dataset_id)
        if last_id is not None:
            skip = last_id + 1
            processed = skip
            passed = self.allowlist.count(dataset_id)
            logger.info(f"[{dataset_id}] Resuming from {skip} ({passed} passed)")
        else:
            skip = None
            processed = 0
            passed = 0
            logger.info(f"[{dataset_id}] Starting from beginning")

        batch: list[Sample] = []
        stream_logger = logging.getLogger(f"pipeline.{dataset_id}")

        for batch in dataset.stream(stream_logger, skip, batch_size=self.batch_size):
            logger.debug(f"[{dataset_id}] Processing batch of {len(batch)} samples")
            passed += self._process_batch(batch, pool)
            processed += len(batch)
            logger.debug(f"[{dataset_id}] {processed} processed, {passed} passed")
            self.checkpoint.update(dataset_id, batch[-1].meta.sample_id)
            batch = []

        self.checkpoint.mark_complete(dataset_id)
        logger.info(f"[{dataset_id}] Complete: {processed} processed, {passed} passed")

    def _process_batch(self, batch: list[Sample], pool: WorkerPool) -> int:
        """Process batch, update allowlist and writer, return count passed."""
        results = pool.process_batch(batch)

        passed_entries = [(r.dataset_id, r.sample_id) for r in results if r.passed]
        if passed_entries:
            self.allowlist.add_batch(passed_entries)

            if self.writer:
                passed_samples = [s for s, r in zip(batch, results) if r.passed]
                self.writer.write_batch(passed_samples)

        return len(passed_entries)
