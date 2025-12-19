from abc import ABC, abstractmethod

from src.schema import RawSample


class BaseFilter(ABC):
    """
    Contract for sample filters.

    Implements logic to determine whether a sample is accepted or rejected.
    """

    @abstractmethod
    def __call__(self, sample: RawSample) -> bool:
        """
        Returns whether the sample passes the filter.
        """
        pass
