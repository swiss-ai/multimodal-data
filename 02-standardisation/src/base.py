from abc import ABC, abstractmethod
from typing import Iterator

from src.schema import Sample


class BaseDataset(ABC):
    """Base class for dataset adapters."""

    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def stream(self, from_id: str | None = None) -> Iterator[Sample]:
        """
        Yield samples. If from_id is provided, skip until that ID
        is found, then yield starting from the next sample.

        Example:
            dataset = ["a", "b", "c", "d"]
            stream(from_id="b") yields "c", "d"
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
