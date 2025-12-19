from abc import ABC, abstractmethod
from typing import Iterator

from src.schema import RawSample


class BaseDataset(ABC):
    """
    Contract for data source adapters.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique id of the data source."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[RawSample]:
        """Yields samples from the data source."""
        pass


class BaseFilter(ABC):
    """
    Contract for sample filters.
    """

    @abstractmethod
    def __call__(self, sample: RawSample) -> bool:
        """Returns whether the sample passes the filter."""
        pass
