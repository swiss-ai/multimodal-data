from abc import ABC, abstractmethod
from typing import Iterator

from src.schema import Sample


class BaseDataset(ABC):
    """Base class for dataset adapters."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this dataset."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """Yield samples one by one."""
        ...


class BaseFilter(ABC):
    """Base class for sample filters."""

    @abstractmethod
    def __call__(self, sample: Sample) -> bool:
        """Return True to keep sample, False to discard."""
        ...
