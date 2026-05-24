from itertools import islice
from pathlib import Path
from types import SimpleNamespace

from loader_recap_datacomp_1b_downloaded_v2 import (
    DATASET_ROOT,
    PROMPT,
    SYSTEM_PROMPT,
    iter_partition_dirs,
)
from loader_recap_datacomp_1b_downloaded_v2 import (
    iter_samples as base_iter_samples,
)
from loader_recap_datacomp_1b_downloaded_v2 import (
    loader as base_loader,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "recap_datacomp_1b_downloaded_v2_sample32"
SAMPLE_LIMIT = 32


def iter_samples(task_id: int, task_count: int):
    partition_dirs = iter_partition_dirs()
    print(
        f"Sample run using completed partitions {partition_dirs[0].name}-{partition_dirs[-1].name} "
        f"under {DATASET_ROOT}; limiting to {SAMPLE_LIMIT} samples"
    )
    yield from islice(base_iter_samples(task_id, task_count), SAMPLE_LIMIT)


loader = SimpleNamespace(
    name="recap_datacomp_1b_downloaded_v2_sample32",
    output_dir=OUTPUT_DIR,
    prompt=PROMPT,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
    batch_size=base_loader.batch_size,
    sampling_params=base_loader.sampling_params,
)
