from abc import ABC, abstractmethod
from typing import Iterator

from src.schema.sample import RawSample, SampleType


class BaseFilter(ABC):
    @abstractmethod
    def process(self, samples: Iterator[RawSample]) -> Iterator[bool]:
        """
        Processes an iterator of samples and yields a boolean for each sample
        indicating whether to keep (True) or discard (False) the sample.
        """
        pass

    @abstractmethod
    def type(self) -> SampleType:
        """
        Returns the type of samples this filter processes.
        """
        pass
