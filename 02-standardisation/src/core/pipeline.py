from collections import defaultdict
from typing import List

from src.adapter.base import BaseAdapter
from src.core.allowlist import AllowlistDB
from src.core.writer import ShardWriter
from src.filter.base import BaseFilter
from src.schema.sample import RawSample, get_sample_type

# TODO: batch processing via multiprocessing


class Pipeline:
    def __init__(self, adapters: List[BaseAdapter], filters: List[BaseFilter]):
        self.adapters = adapters
        self.filters = filters

        self.meta_filters = defaultdict(list)
        self.content_filters = defaultdict(list)

        for f in filters:
            if f.requires_content:
                self.content_filters[f.sample_type].append(f)
            else:
                self.meta_filters[f.sample_type].append(f)

    def scan(self, allowlist_path: str, batch_size: int):
        """
        Iterates adapters, applies filters, and populates the Allowlist.
        """
        with AllowlistDB(allowlist_path) as allowlist:
            for adapter in self.adapters:
                batch_buffer = []

                for sample in adapter.stream():
                    s_type = get_sample_type(sample)

                    # meta filters
                    if self.meta_filters[s_type]:
                        if not self.apply_filters(sample, self.meta_filters[s_type]):
                            continue

                    # content filters
                    if self.content_filters[s_type]:
                        sample = adapter.hydrate(sample)
                        if not self.apply_filters(sample, self.content_filters[s_type]):
                            continue

                    batch_buffer.append(
                        (sample.meta.dataset_name, sample.meta.sample_id)
                    )

                    if len(batch_buffer) >= batch_size:
                        allowlist.add_batch(batch_buffer)
                        batch_buffer = []

                # flush remaining
                if batch_buffer:
                    allowlist.add_batch(batch_buffer)

    def build(self, allowlist_path: str, output_dir: str):
        """
        Iterates adapters (again), checks Allowlist, hydrates, and writes WebDatasets.
        """
        with AllowlistDB(allowlist_path) as allowlist:
            for adapter in self.adapters:
                shard_name = f"{adapter.name}_part"

                with ShardWriter(output_dir, shard_name) as writer:
                    for sample in adapter.stream():
                        if not allowlist.exists(adapter.name, sample.meta.sample_id):
                            continue

                        full_sample = adapter.hydrate(sample)
                        writer.write(full_sample)

    def apply_filters(self, sample: RawSample, filters: List[BaseFilter]) -> bool:
        for f in filters:
            if not f(sample):
                return False
        return True
