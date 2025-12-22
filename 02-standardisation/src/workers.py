import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable

# import ray.util.multiprocessing as mp
from src.base import BaseFilter
from src.schema import Sample

logger = logging.getLogger("pipeline.workers")


@dataclass
class FilterResult:
    dataset_id: str
    sample_id: str
    passed: bool


_worker_filters: list[BaseFilter] | None = None


def _init_worker(filter_factories: list[Callable[[], BaseFilter]]):
    global _worker_filters
    _worker_filters = [factory() for factory in filter_factories]


def _process_sample(data: bytes) -> FilterResult:
    global _worker_filters
    sample = Sample.deserialize(data)

    try:
        assert _worker_filters is not None, "Filters not initialized"
        passed = all(f(sample) for f in _worker_filters)

    except Exception as e:
        logger.exception(f"Error processing sample: {e}")
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
        filter_factories: list[Callable[[], BaseFilter]],
        num_workers: int,
    ):
        logger.info(f"Starting worker pool with {num_workers} workers")
        self.num_workers = num_workers
        self.pool = mp.Pool(
            processes=self.num_workers,
            initializer=_init_worker,
            initargs=(filter_factories,),
        )

    def process_batch(self, samples: list[Sample]) -> list[FilterResult]:
        serialized = [s.serialize() for s in samples]
        return list(self.pool.map(_process_sample, serialized))

    def close(self):
        logger.info("Shutting down worker pool")
        self.pool.close()
        self.pool.join()

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *_):
        self.close()
