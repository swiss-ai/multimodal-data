import logging
from collections.abc import Callable, Sequence

from src.allowlist import Allowlist
from src.base import BaseDataset, BaseFilter
from src.checkpoint import Checkpoint
from src.schema import Sample
from src.workers import WorkerPool

logger = logging.getLogger("pipeline")


class Pipeline:
    """Scans datasets, applies filters, builds manifest."""

    def __init__(
        self,
        dataset_factories: Sequence[Callable[[], BaseDataset]],
        filter_factories: Sequence[Callable[[], BaseFilter]],
        data_dir: str,
        num_workers: int,
        batch_size: int,
    ):
        self.dataset_factories = dataset_factories
        self.filter_factories = filter_factories
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.allowlist = Allowlist(f"{data_dir}/manifest.db")
        self.checkpoint = Checkpoint(f"{data_dir}/checkpoint.db")

    def scan(self):
        logger.info(f"Starting scan of {len(self.dataset_factories)} dataset(s)")
        with WorkerPool(self.filter_factories, self.num_workers) as pool:
            for factory in self.dataset_factories:
                self._scan_dataset(factory(), pool)
        logger.info("Scan complete")

    def _scan_dataset(self, dataset: BaseDataset, pool: WorkerPool):
        dataset_id = dataset.id

        if self.checkpoint.is_complete(dataset_id):
            logger.info(f"[{dataset_id}] Skipping (already complete)")
            return

        from_id = self.checkpoint.get_resume_point(dataset_id)
        if from_id:
            logger.info(f"[{dataset_id}] Resuming from sample {from_id}")
        else:
            logger.info(f"[{dataset_id}] Starting")

        batch: list[Sample] = []
        processed, passed = 0, 0

        for sample in dataset.stream(from_id):
            batch.append(sample)

            if len(batch) >= self.batch_size:
                p = self._process_batch(batch, pool)
                processed += len(batch)
                passed += p
                self.checkpoint.update(dataset_id, batch[-1].meta.sample_id)
                logger.debug(f"[{dataset_id}] {processed} processed, {passed} passed")
                batch = []

        # remaining samples
        if batch:
            p = self._process_batch(batch, pool)
            processed += len(batch)
            passed += p
            self.checkpoint.update(dataset_id, batch[-1].meta.sample_id)

        self.checkpoint.mark_complete(dataset_id)
        logger.info(f"[{dataset_id}] Complete: {processed} processed, {passed} passed")

    def _process_batch(self, batch: list[Sample], pool: WorkerPool) -> int:
        """Process batch, update allowlist, return count passed."""
        results = pool.process_batch(batch)
        passed_entries = [(r.dataset_id, r.sample_id) for r in results if r.passed]
        if passed_entries:
            self.allowlist.add_batch(passed_entries)
        return len(passed_entries)
