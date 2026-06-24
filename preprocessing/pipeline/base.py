from abc import ABC, abstractmethod
from typing import Iterator

from pipeline.schema import Sample


class BaseDataset(ABC):
    """Base class for dataset adapters."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def stream(
        self, logger, skip: int | None = None, batch_size: int = 1
    ) -> Iterator[list[Sample]]:
        """
        Yield batches of samples. If 'skip' is provided, skip the first 'skip' samples.

        Example:
            dataset = ["a", "b", "c", "d"]
            stream(skip=2)               yields ["c"], ["d"]
            stream(skip=2, batch_size=2) yields ["c" ,  "d"]
        """
        ...


class BaseFilter(ABC):
    """Base class for sample filters."""

    @abstractmethod
    def process_batch(self, samples: list[Sample]) -> list[Sample]:
        """
        Process a batch of samples. Return only the samples that pass the filter.
        Filters can modify samples (e.g., downsample images) before returning them.
        Can be stateful, but must be thread-safe.
        """
        ...


class BaseWriter(ABC):
    """Base class for pipeline output writers."""

    @abstractmethod
    def open(self, dataset_id: str):
        """Called before processing a dataset."""
        ...

    @abstractmethod
    def write_batch(self, samples: list[Sample]):
        """Write a batch of accepted samples."""
        ...

    @abstractmethod
    def close(self):
        """Called after processing completes."""
        ...
