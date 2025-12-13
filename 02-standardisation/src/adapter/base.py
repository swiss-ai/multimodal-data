from abc import ABC, abstractmethod
from typing import Iterator

from src.schema.sample import RawSample, SampleType


class BaseAdapter(ABC):
    """
    Contract for data source adapters in the data processing pipeline.
    """

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
        This is only called by the pipeline if the sample passes metadata filtering
        or is pre-determined to require full content.
        """
        pass

    @abstractmethod
    def type(self) -> SampleType:
        """
        Returns the type of samples this adapter produces.
        """
        pass
