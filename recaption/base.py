from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterator


class Loader(ABC):
    def __init__(self, task_id: int, task_count: int):
        self.task_id = task_id
        self.task_count = task_count

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[str, list[str]]]:
        """Yield (sample_id, [b64_img, ...]) for each sample."""

    def stream(self, bs: int) -> Iterator[list[tuple[str, list[str]]]]:
        """Yield batches of up to *bs* samples."""
        it = iter(self)
        while batch := list(islice(it, bs)):
            yield batch
