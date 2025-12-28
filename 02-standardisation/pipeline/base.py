from abc import ABC, abstractmethod
from typing import Iterator

from pipeline.schema import Sample


class BaseDataset(ABC):
    """Base class for dataset adapters."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def stream(self, logger, skip: int | None = None) -> Iterator[Sample]:
        """
        Yield samples. If 'skip' is provided, skip the first 'skip' samples.

        Example:
            dataset = ["a", "b", "c", "d"]
            stream(skip=2) yields "c", "d"
        """
        ...


class BaseFilter(ABC):
    """Base class for sample filters."""

    @abstractmethod
    def process_batch(self, samples: list[Sample]) -> list[bool]:
        """
        Process a batch of samples. Return list of bools (True=keep, False=discard).
        Can be stateful, but must be thread-safe.
        """
        ...


class BaseSink(ABC):
    """Base class for pipeline output sinks."""

    def open(self) -> None:
        """Called before processing starts."""
        pass

    @abstractmethod
    def write_batch(self, samples: list[Sample]) -> None:
        """Write a batch of accepted samples."""
        ...

    def close(self) -> None:
        """Called after processing completes."""
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
