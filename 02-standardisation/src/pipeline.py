import logging
from typing import Callable

from src.allowlist import Allowlist
from src.base import BaseDataset, BaseFilter
from src.checkpoint import Checkpoint
from src.schema import Sample
from src.workers import WorkerPool

logger = logging.getLogger()


class Pipeline:
    """Scans datasets and applies filters to build the manifest of approved samples."""

    def __init__(
        self,
        datasets: list[BaseDataset],
        filter_factories: list[Callable[[], BaseFilter]],
        data_dir: str,
        num_workers: int,
        batch_size: int,
    ):
        self.datasets = datasets
        self.filter_factories = filter_factories

        self.data_dir = data_dir
        self.allowlist = Allowlist(f"{self.data_dir}/manifest.db")
        self.checkpoint = Checkpoint(f"{self.data_dir}/checkpoint.db")

        self.num_workers = num_workers
        self.batch_size = batch_size

    def scan(self):
        logger.info("Starting scan")

        with WorkerPool(self.filter_factories, self.num_workers) as pool:
            for dataset in self.datasets:
                self._scan_dataset(dataset, pool)

        logger.info("Scan complete")

    def _scan_dataset(self, dataset: BaseDataset, pool: WorkerPool):
        dataset_id = dataset.id

        last_sample = self.checkpoint.get_resume_point(dataset_id)
        if last_sample is True:
            logger.info(f"Skipping completed: {dataset_id}")
            return
        elif last_sample is False:
            resume_from = None
        else:
            resume_from = last_sample
            logger.info(f"Resuming {dataset_id} from {resume_from}")

        logger.info(f"Scanning: {dataset_id}")

        batch: list[Sample] = []
        processed, passed = 0, 0
        skipping = resume_from is not None

        for sample in dataset:
            if skipping:
                if sample.meta.sample_id == resume_from:
                    skipping = False
                continue

            batch.append(sample)

            if len(batch) >= self.batch_size:
                p = self._process_batch(batch, pool)
                processed += len(batch)
                passed += p
                self.checkpoint.update(dataset_id, batch[-1].meta.sample_id)
                logger.info(f"[{dataset_id}] {processed} processed, {passed} passed")
                batch = []

        if batch:
            p = self._process_batch(batch, pool)
            processed += len(batch)
            passed += p
            self.checkpoint.update(dataset_id, batch[-1].meta.sample_id)
            logger.info(f"[{dataset_id}] {processed} processed, {passed} passed")

        self.checkpoint.mark_complete(dataset_id)
        logger.info(f"Completed {dataset_id}: {processed} processed, {passed} passed")

    def _process_batch(self, batch: list[Sample], pool: WorkerPool) -> int:
        """Filter batch and update allowlist. Returns count passed."""
        results = pool.process_batch(batch)
        passed_entries = [(r.dataset_id, r.sample_id) for r in results if r.passed]
        if passed_entries:
            self.allowlist.add_batch(passed_entries)
        return len(passed_entries)
