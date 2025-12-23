from abc import ABC, abstractmethod
from typing import Iterator

from pipeline.schema import Sample


class BaseDataset(ABC):
    """Base class for dataset adapters."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def stream(self, skip: int | None = None) -> Iterator[Sample]:
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
    def __call__(self, sample: Sample) -> bool:
        """
        Return True to keep sample, False to discard.

        This method can be stateful but must be thread-safe, i.e., it should work
        correctly even when called from multiple threads simultaneously.

        Exceptions are logged and treated as filter failures (i.e., return False).
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
