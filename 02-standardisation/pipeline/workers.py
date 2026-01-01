import logging
import multiprocessing as mp
from collections.abc import Callable, Sequence
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


FilterFactory = Callable[[], BaseFilter]

_worker_filters: list[BaseFilter] | None = None


def _init_worker(filter_factories: Sequence[FilterFactory]):
    global _worker_filters
    _worker_filters = [f() for f in filter_factories]


def _process_batch(batch_data: list[Sample]) -> list[FilterResult]:
    global _worker_filters
    assert _worker_filters is not None, "Filters not initialized"

    # samples = [Sample.deserialize(data) for data in batch_data]  # ray version
    samples = batch_data
    results = []

    passed = [True] * len(samples)
    for f in _worker_filters:
        try:
            filter_results = f.process_batch(samples)
            passed = [p and r for p, r in zip(passed, filter_results)]
        except Exception:
            did, sid = samples[0].meta.dataset_id, samples[0].meta.sample_id
            logger.exception(f"Error filtering batch of {did}/{sid}, marking as failed")
            passed = [False] * len(samples)
            break

    for sample, p in zip(samples, passed):
        results.append(
            FilterResult(
                dataset_id=sample.meta.dataset_id,
                sample_id=sample.meta.sample_id,
                passed=p,
            )
        )

    return results


class WorkerPool:
    def __init__(
        self,
        filter_factories: Sequence[FilterFactory],
        num_workers: int,
    ):
        self.num_workers = num_workers
        logger.info(f"Starting {num_workers} workers")
        self.pool = mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(filter_factories,),
        )
        logger.debug("Worker pool ready")

    def process_batch(self, samples: list[Sample]) -> list[FilterResult]:
        # serialized = [s.serialize() for s in samples]
        serialized = samples

        sub_batch_size = len(serialized) // self.num_workers
        sub_batches = [
            serialized[i * sub_batch_size : (i + 1) * sub_batch_size]
            for i in range(self.num_workers - 1)
        ]
        sub_batches.append(serialized[(self.num_workers - 1) * sub_batch_size :])

        logger.debug(
            f"Processing batch of {len(samples)} samples in "
            f"{len(sub_batches)} sub-batches"
        )

        nested_results = self.pool.map(_process_batch, sub_batches)
        return [r for results in nested_results for r in results]

    def close(self):
        logger.debug("Shutting down workers")
        self.pool.close()
        self.pool.join()
        logger.info("Workers stopped")

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *_):
        self.close()
