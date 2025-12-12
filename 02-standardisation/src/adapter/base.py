from abc import ABC, abstractmethod
from typing import Iterator

from src.schema.sample import RawSample, SampleType


class BaseAdapter(ABC):
    @abstractmethod
    def stream(self) -> Iterator[RawSample]:
        """
        Yields standardized, strictly typed samples.
        """
        pass

    @abstractmethod
    def type(self) -> SampleType:
        """
        Returns the type of samples this adapter produces.
        """
        pass
