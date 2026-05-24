#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___UCSC-VLAA___Recap-DataComp-1B___downloaded")
CAPTIONS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "recap_datacomp_1b_downloaded_v2"
MIN_PARTITION = 0
MAX_PARTITION = 1067
LOGICAL_TASK_COUNT = 1024
CAPTION_FILE_RE = re.compile(r"captions_task(\d{4})\.jsonl$")
IMAGE_KEYS_TO_EXTENSIONS = ("jpg", "jpeg", "png", "webp")


@dataclass(frozen=True)
class SampleRef:
    sample_id: str
    source_tar: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--direct-checks", type=int, default=256)
    return parser.parse_args()


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
        raise FileNotFoundError(f"No completed partitions found under {DATASET_ROOT}")
    return partition_dirs


def iter_shard_paths() -> list[Path]:
    shard_paths: list[Path] = []
    for partition_dir in iter_partition_dirs():
        shard_paths.extend(sorted((partition_dir / "chunk_000").glob("*.tar")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard tar files found under {DATASET_ROOT}")
    return shard_paths


def assigned_shards(shard_paths: list[Path], task_id: int, task_count: int) -> list[Path]:
    total = len(shard_paths)
    return shard_paths[total * task_id // task_count : total * (task_id + 1) // task_count]


def build_sample_id(source_tar_path: Path, source_key: str) -> str:
    partition_name = source_tar_path.parent.parent.name
    shard_name = source_tar_path.stem
    return f"{partition_name}__{shard_name}__{source_key}"


def caption_paths() -> list[Path]:
    paths = sorted(CAPTIONS_ROOT.glob("captions_task*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No caption files found under {CAPTIONS_ROOT}")
    return paths


def parse_task_id(path: Path) -> int:
    match = CAPTION_FILE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected caption filename: {path.name}")
    return int(match.group(1))


def validate_caption_file(
    caption_path_str: str,
    task_id: int,
    allowed_tar_list: list[str],
) -> tuple[int, int, int, list[SampleRef]]:
    caption_path = Path(caption_path_str)
    allowed_tars = set(allowed_tar_list)
    if not allowed_tars:
        raise RuntimeError(f"Task {task_id} has no assigned shards")

    seen_sample_ids: set[str] = set()
    seen_source_tars: set[str] = set()
    sample_refs: list[SampleRef] = []
    local_count = 0
    rng = random.Random(task_id)

    with caption_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            caption = payload.get("caption")
            metadata = payload.get("metadata") or {}
            source_tar = metadata.get("source_tar")
            if not isinstance(sample_id, str) or not sample_id:
                raise RuntimeError(f"{caption_path}:{line_number} missing sample_id")
            if not isinstance(caption, str) or not caption.strip():
                raise RuntimeError(f"{caption_path}:{line_number} missing caption text")
            if not isinstance(source_tar, str) or not source_tar:
                raise RuntimeError(f"{caption_path}:{line_number} missing source_tar")
            if source_tar not in allowed_tars:
                raise RuntimeError(f"{caption_path}:{line_number} references shard outside task slice: {source_tar}")
            if sample_id in seen_sample_ids:
                raise RuntimeError(f"{caption_path}:{line_number} duplicate sample_id within task: {sample_id}")

            partition_name, shard_name, source_key = sample_id.split("__", 2)
            source_tar_path = Path(source_tar)
            if partition_name != source_tar_path.parent.parent.name:
                raise RuntimeError(f"{caption_path}:{line_number} partition mismatch for {sample_id}")
            if shard_name != source_tar_path.stem:
                raise RuntimeError(f"{caption_path}:{line_number} shard mismatch for {sample_id}")
            if not source_key:
                raise RuntimeError(f"{caption_path}:{line_number} empty source key")

            seen_sample_ids.add(sample_id)
            seen_source_tars.add(source_tar)
            local_count += 1

            # Keep a tiny per-file reservoir; enough for a broad direct-lookup sample later.
            if len(sample_refs) < 4:
                sample_refs.append(SampleRef(sample_id=sample_id, source_tar=source_tar))
            else:
                index = rng.randrange(local_count)
                if index < len(sample_refs):
                    sample_refs[index] = SampleRef(sample_id=sample_id, source_tar=source_tar)

    if local_count == 0:
        raise RuntimeError(f"{caption_path} is empty")

    return task_id, local_count, len(seen_source_tars), sample_refs


def validate_caption_files(worker_count: int) -> tuple[dict[str, int], list[SampleRef]]:
    shard_paths = iter_shard_paths()
    all_caption_paths = caption_paths()
    if len(all_caption_paths) != LOGICAL_TASK_COUNT:
        raise RuntimeError(f"Expected {LOGICAL_TASK_COUNT} caption files, found {len(all_caption_paths)}")

    worker_count = max(1, min(worker_count, len(all_caption_paths)))
    total_records = 0
    total_source_tars = 0
    sample_refs: list[SampleRef] = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for caption_path in all_caption_paths:
            task_id = parse_task_id(caption_path)
            allowed_paths = assigned_shards(shard_paths, task_id, LOGICAL_TASK_COUNT)
            futures.append(
                executor.submit(
                    validate_caption_file,
                    str(caption_path),
                    task_id,
                    [str(path) for path in allowed_paths],
                )
            )

        for future in as_completed(futures):
            task_id, local_count, source_tar_count, local_refs = future.result()
            total_records += local_count
            total_source_tars += source_tar_count
            sample_refs.extend(local_refs)
            print(
                f"[task {task_id:04d}] validated {local_count} records across {source_tar_count} source tars",
                flush=True,
            )

    return {
        "caption_files": len(all_caption_paths),
        "completed_partitions": len(iter_partition_dirs()),
        "records": total_records,
        "source_tars_referenced": total_source_tars,
        "task_count": LOGICAL_TASK_COUNT,
        "total_shards": len(shard_paths),
    }, sample_refs


def validate_single_lookup(ref: SampleRef) -> int:
    source_tar_path = Path(ref.source_tar)
    expected_partition, expected_shard, expected_key = ref.sample_id.split("__", 2)
    matches = 0
    dataset = wds.WebDataset(str(source_tar_path), shardshuffle=False, empty_check=False)
    for sample in dataset:
        source_key = sample["__key__"]
        if source_key != expected_key:
            continue
        matches += 1
        rebuilt = build_sample_id(source_tar_path, source_key)
        if rebuilt != ref.sample_id:
            raise RuntimeError(f"Direct lookup rebuilt wrong sample id: {rebuilt} != {ref.sample_id}")
        metadata = json.loads(bytes(sample.get("json", b"{}")).decode("utf-8"))
        metadata_key = metadata.get("key")
        if metadata_key is not None and str(metadata_key) != source_key:
            raise RuntimeError(f"Metadata key mismatch in {source_tar_path}: {metadata_key!r} != {source_key!r}")
        if source_tar_path.parent.parent.name != expected_partition:
            raise RuntimeError(f"Partition mismatch during direct lookup for {ref.sample_id}")
        if source_tar_path.stem != expected_shard:
            raise RuntimeError(f"Shard mismatch during direct lookup for {ref.sample_id}")
        if not any(sample.get(key) is not None for key in IMAGE_KEYS_TO_EXTENSIONS):
            raise RuntimeError(f"No image payload found for {ref.sample_id}")
        break
    if matches != 1:
        raise RuntimeError(f"Expected exactly one match for {ref.sample_id} in {source_tar_path}, got {matches}")
    return 1


def validate_direct_lookup(
    sample_refs: list[SampleRef],
    checks: int = 256,
    worker_count: int = 32,
) -> dict[str, int]:
    if not sample_refs:
        raise RuntimeError("No sample references available for direct lookup validation")

    rng = random.Random(1)
    if checks >= len(sample_refs):
        chosen = sample_refs
    else:
        chosen = rng.sample(sample_refs, checks)

    worker_count = max(1, min(worker_count, len(chosen)))
    verified = 0
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(validate_single_lookup, ref) for ref in chosen]
        for future in as_completed(futures):
            verified += future.result()

    return {"direct_lookup_checks": verified}


def main() -> None:
    args = parse_args()
    structural_summary, sample_refs = validate_caption_files(args.workers)
    direct_summary = validate_direct_lookup(
        sample_refs,
        checks=args.direct_checks,
        worker_count=min(32, args.workers),
    )
    summary = {**structural_summary, **direct_summary}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
