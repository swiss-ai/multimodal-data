from abc import ABC, abstractmethod

from src.schema.sample import RawSample, SampleType


class BaseFilter(ABC):
    """
    Contract for sample filters.

    Implements logic to determine whether a sample is accepted or rejected.
    """

    @property
    @abstractmethod
    def requires_content(self) -> bool:
        """
        Filter must specify whether they need full content to operate.

        True if the filter needs full content (e.g., image bytes, full text),
        False if it can operate on metadata alone.
        """
        pass

    @property
    @abstractmethod
    def sample_type(self) -> SampleType:
        """
        SampleType this filter targets.
        """
        pass

    @abstractmethod
    def __call__(self, samples: RawSample) -> bool:
        """
        Returns whether the sample passes the filter.
        """
        pass
