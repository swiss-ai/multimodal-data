from abc import ABC, abstractmethod
from typing import Iterator

from src.schema.sample import RawSample, SampleType


class BaseFilter(ABC):
    """
    Contract for sample filters in the data processing pipeline.
    Filters determine whether samples should be kept or discarded based
    on metadata and/or full content.
    """

    @property
    @abstractmethod
    def requires_content(self) -> bool:
        """
        Filter must specify whether they need full content to operate.

        True if the filter needs full content (e.g., image bytes, full text),
        False if it can operate on metadata alone.
        """
        return False

    @abstractmethod
    def process(self, samples: RawSample) -> bool:
        """
        Returns whether the sample passes the filter.
        """
        pass

    @abstractmethod
    def type(self) -> SampleType:
        """
        Returns the type of samples this filter processes.
        """
        pass
