from abc import ABC, abstractmethod
from typing import Iterator

from src.schema.sample import RawSample


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
        Yields lightweight samples (metadata) from the data source.
        Fields like 'image' or 'text' can be None at this stage.
        """
        pass

    @abstractmethod
    def hydrate(self, sample: RawSample) -> RawSample:
        """
        Populates the heavy data (image bytes, full text) for a given sample.
        """
        pass
