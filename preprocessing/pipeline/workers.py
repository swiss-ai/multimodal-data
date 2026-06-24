import logging
import multiprocessing as mp
from collections.abc import Callable, Sequence

# import ray.util.multiprocessing as mp
from pipeline.base import BaseFilter
from pipeline.schema import Sample

logger = logging.getLogger("pipeline.workers")

FilterFactory = Callable[[], BaseFilter]

_worker_filters: list[BaseFilter] | None = None


def _init_worker(filter_factories: Sequence[FilterFactory]):
    global _worker_filters
    _worker_filters = [f() for f in filter_factories]


def _process_batch(batch_data: list[Sample]) -> list[Sample]:
    global _worker_filters
    assert _worker_filters is not None, "Filters not initialized"

    # samples = [Sample.deserialize(data) for data in batch_data]  # ray version
    current = list(batch_data)

    for f in _worker_filters:
        try:
            current = f.process_batch(current)
        except Exception:
            if batch_data:
                did, sid = batch_data[0].meta.dataset_id, batch_data[0].meta.sample_id
                logger.exception(f"Error filtering batch of {did}/{sid}, mark failed")
            return []

    return current


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

    def process_batch(self, samples: list[Sample]) -> list[Sample]:
        if not samples:
            return []

        # serialized = [s.serialize() for s in samples]
        serialized = samples

        sub_batch_size = max(1, len(serialized) // self.num_workers)
        sub_batches = [
            serialized[i : i + sub_batch_size]
            for i in range(0, len(serialized), sub_batch_size)
        ]

        logger.debug(
            f"Processing batch of {len(samples)} samples in "
            f"{len(sub_batches)} sub-batches"
        )

        nested_results = self.pool.map(_process_batch, sub_batches)
        return [s for batch in nested_results for s in batch]

    def close(self):
        logger.debug("Shutting down workers")
        self.pool.close()
        self.pool.join()
        logger.info("Workers stopped")

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *_):
        self.close()
