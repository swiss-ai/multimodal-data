from collections import defaultdict
from typing import Iterator, List, Set

from src.adapter.base import BaseAdapter
from src.filter.base import BaseFilter
from src.schema.sample import RawSample, get_sample_type


class Pipeline:
    def __init__(self, adapters: List[BaseAdapter], filters: List[BaseFilter]):
        self.adapters = adapters

        self.meta_filters = defaultdict(list)
        self.content_filters = defaultdict(list)

        for f in filters:
            target_type = f.sample_type()
            if f.requires_content:
                self.content_filters[target_type].append(f)
            else:
                self.meta_filters[target_type].append(f)

    def stream_filter(self) -> Iterator[RawSample]:
        """
        Stream samples through filtering pipeline with lazy hydration.
        """
        for adapter in self.adapters:
            for sample in adapter.stream():
                sample_type = get_sample_type(sample)
                # meta filtering
                if not self.apply_filters(sample, self.meta_filters[sample_type]):
                    continue
                # full content filtering
                sample = adapter.hydrate(sample)
                if not self.apply_filters(sample, self.content_filters[sample_type]):
                    continue
                yield sample

    def stream_manifest(self, manifest: Set[str]) -> Iterator[RawSample]:
        """
        Stream samples based on a predefined whitelist (manifest).
        """
        for adapter in self.adapters:
            for sample in adapter.stream():
                key = f"{sample.meta.dataset_name}.{sample.meta.sample_id}"
                if key in manifest:
                    yield adapter.hydrate(sample)

    def apply_filters(self, sample: RawSample, filters: List[BaseFilter]) -> bool:
        """
        Applies a list of filters to a single sample.
        """
        for f in filters:
            if not f.process(sample):
                return False
        return True
