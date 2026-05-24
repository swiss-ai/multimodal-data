from pathlib import Path
from types import SimpleNamespace

from loader_openimages_v7_dense___full_v2 import (
    DATASET_ROOT,
    PROMPT,
    SYSTEM_PROMPT,
    base_iter_shard_paths,
    iter_samples_from_shard,
)
from loader_openimages_v7_dense___full_v2 import (
    loader as base_loader,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "openimages_v7_dense___full_v2_sample"
SAMPLE_SHARD_LIMIT = 4


def iter_samples(task_id: int, task_count: int):
    shard_paths = base_iter_shard_paths()[:SAMPLE_SHARD_LIMIT]
    total = len(shard_paths)
    assigned_paths = shard_paths[total * task_id // task_count : total * (task_id + 1) // task_count]
    print(
        f"Sample run using {len(shard_paths)} shards under {DATASET_ROOT}; "
        f"task {task_id}/{task_count - 1} assigned {len(assigned_paths)}"
    )
    if assigned_paths:
        print(f"Assigned first shard: {assigned_paths[0]}")
        print(f"Assigned last shard:  {assigned_paths[-1]}")

    for shard_path in assigned_paths:
        yield from iter_samples_from_shard(shard_path)


loader = SimpleNamespace(
    name="openimages_v7_dense___full_v2_sample",
    output_dir=OUTPUT_DIR,
    prompt=PROMPT,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
    batch_size=base_loader.batch_size,
    sampling_params=base_loader.sampling_params,
)
