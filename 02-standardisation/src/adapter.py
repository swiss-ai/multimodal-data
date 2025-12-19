from abc import ABC, abstractmethod
from typing import Iterator

from src.schema import RawSample


class BaseAdapter(ABC):
    """
    Contract for data source adapters.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name of the adapter.
        """
        pass

    @abstractmethod
    def stream(self) -> Iterator[RawSample]:
        """
        Yields samples from the data source.
        """
        pass
