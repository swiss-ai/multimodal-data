import logging
import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import dataclass

# import ray.util.multiprocessing as mp
from pipeline.base import BaseFilter
from pipeline.schema import Sample

logger = logging.getLogger("pipeline.workers")


@dataclass
class FilterResult:
    dataset_id: str
    sample_id: int
    passed: bool


_worker_filters: list[BaseFilter] | None = None


def _init_worker(filters: list[BaseFilter]):
    global _worker_filters
    _worker_filters = filters


def _process_sample(data: bytes) -> FilterResult:
    global _worker_filters
    sample = Sample.deserialize(data)

    try:
        assert _worker_filters is not None, "Filters not initialized"
        passed = all(f(sample) for f in _worker_filters)

    except Exception:
        did, sid = sample.meta.dataset_id, sample.meta.sample_id
        logger.exception(f"Error processing sample {did}/{sid}, marking as failed")
        return FilterResult(
            dataset_id=sample.meta.dataset_id,
            sample_id=sample.meta.sample_id,
            passed=False,
        )

    return FilterResult(
        dataset_id=sample.meta.dataset_id,
        sample_id=sample.meta.sample_id,
        passed=passed,
    )


class WorkerPool:
    def __init__(
        self,
        filters: Sequence[BaseFilter],
        num_workers: int,
    ):
        self.num_workers = num_workers
        logger.info(f"Starting {num_workers} workers")
        self.pool = mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(filters,),
        )
        logger.debug("Worker pool ready")

    def process_batch(self, samples: list[Sample]) -> list[FilterResult]:
        serialized = [s.serialize() for s in samples]
        return list(self.pool.map(_process_sample, serialized))

    def close(self):
        logger.debug("Shutting down workers")
        self.pool.close()
        self.pool.join()
        logger.info("Workers stopped")

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *_):
        self.close()
