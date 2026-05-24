import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import webdataset as wds
from loader_openimages_v7_dense___full_v2 import PROMPT, SYSTEM_PROMPT
from loader_spawning_pd12m_full_v2 import extract_image, sanitize_image_if_needed
from vllm import SamplingParams

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___UCSC-VLAA___Recap-DataComp-1B___downloaded")
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "recap_datacomp_1b_downloaded_v2"
MIN_PARTITION = 0
MAX_PARTITION = 1067


def iter_partition_dirs() -> list[Path]:
    partition_dirs = []
    for path in sorted(DATASET_ROOT.iterdir()):
        if not path.is_dir() or not path.name.isdigit() or len(path.name) != 5:
            continue
        part_id = int(path.name)
        if part_id < MIN_PARTITION or part_id > MAX_PARTITION:
            continue
        if not (path / "_SUCCESS").exists():
            continue
        if not (path / "chunk_000" / "_SUCCESS").exists():
            continue
        partition_dirs.append(path)

    if not partition_dirs:
        raise FileNotFoundError(
            f"No completed partition directories found under {DATASET_ROOT} "
            f"for range {MIN_PARTITION:05d}-{MAX_PARTITION:05d}"
        )
    return partition_dirs


def iter_shard_paths() -> list[Path]:
    shard_paths = []
    for partition_dir in iter_partition_dirs():
        shard_paths.extend(sorted((partition_dir / "chunk_000").glob("*.tar")))

    if not shard_paths:
        raise FileNotFoundError(f"No shard tars found under {DATASET_ROOT}")
    return shard_paths


def iter_samples_from_shard(shard_path: Path) -> Iterator[dict]:
    dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)
    partition_name = shard_path.parent.parent.name
    shard_name = shard_path.stem

    for sample in dataset:
        image_bytes, media_type = extract_image(sample)
        raw_metadata = sample.get("json", b"{}")
        metadata = json.loads(bytes(raw_metadata).decode("utf-8"))
        source_key = sample["__key__"]
        metadata_key = metadata.get("key")
        if metadata_key is not None and str(metadata_key) != source_key:
            raise ValueError(f"Mismatched key for {shard_path}: sample={source_key!r} metadata={metadata_key!r}")

        sample_id = f"{partition_name}__{shard_name}__{source_key}"
        image_bytes, media_type = sanitize_image_if_needed(
            image_bytes,
            media_type,
            sample_id,
        )

        source_caption = sample.get("txt")
        if source_caption is not None:
            source_caption = bytes(source_caption).decode("utf-8").strip()

        yield {
            "sample_id": sample_id,
            "image_bytes": image_bytes,
            "media_type": media_type,
            "metadata": {
                **metadata,
                "source_key": source_key,
                "source_caption": source_caption,
                "partition": partition_name,
                "shard": shard_path.name,
                "source_tar": sample.get("__url__", str(shard_path)),
            },
        }


def iter_samples(task_id: int, task_count: int) -> Iterator[dict]:
    shard_paths = iter_shard_paths()
    total = len(shard_paths)
    assigned_paths = shard_paths[total * task_id // task_count : total * (task_id + 1) // task_count]
    partition_count = len(iter_partition_dirs())
    print(
        f"Discovered {partition_count} completed partitions and {len(shard_paths)} shards "
        f"under {DATASET_ROOT}; task {task_id}/{task_count - 1} assigned {len(assigned_paths)}"
    )
    if assigned_paths:
        print(f"Assigned first shard: {assigned_paths[0]}")
        print(f"Assigned last shard:  {assigned_paths[-1]}")

    for shard_path in assigned_paths:
        yield from iter_samples_from_shard(shard_path)


loader = SimpleNamespace(
    name="recap_datacomp_1b_downloaded_v2",
    output_dir=OUTPUT_DIR,
    prompt=PROMPT,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
    batch_size=64,
    sampling_params=SamplingParams(
        temperature=0.1,
        top_p=0.9,
        top_k=20,
        repetition_penalty=1.2,
        max_tokens=768,
    ),
)
