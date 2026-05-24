#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___UCSC-VLAA___Recap-DataComp-1B___downloaded")
CAPTIONS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "recap_datacomp_1b_downloaded_v2"
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/vision-datasets/hf___UCSC-VLAA___Recap-DataComp-1B___downloaded___recap")
IMAGE_KEYS_TO_EXTENSIONS = {
    "jpg": "jpg",
    "jpeg": "jpeg",
    "png": "png",
    "webp": "webp",
}
MIN_PARTITION = 0
MAX_PARTITION = 1067
SHARD_MAXCOUNT = 100_000
SHARD_MAXSIZE = 8_000_000_000
PROGRESS_EVERY = 100_000
TEXT_METADATA_KEYS = {
    "caption",
    "captions",
    "description",
    "prompt",
    "recaption",
    "re_caption",
    "text",
    "texts",
    "txt",
}


@dataclass(frozen=True)
class CaptionRecord:
    sample_id: str
    caption: str
    source_tar: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subdir", default=".")
    parser.add_argument("--workers", type=int, default=256)
    parser.add_argument("--task-ids", help="Comma-separated logical task ids to export")
    parser.add_argument("--limit-total", type=int)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--shard-maxcount", type=int, default=SHARD_MAXCOUNT)
    parser.add_argument("--shard-maxsize", type=int, default=SHARD_MAXSIZE)
    return parser.parse_args()


def resolve_output_dir(output_root: Path, subdir: str) -> Path:
    if subdir in {"", "."}:
        return output_root
    return output_root / subdir


def parse_task_ids(value: str | None) -> list[int] | None:
    if value is None:
        return None
    task_ids = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        task_ids.append(int(chunk))
    return sorted(set(task_ids))


def list_caption_paths(task_ids: list[int] | None) -> list[Path]:
    if task_ids is None:
        paths = sorted(CAPTIONS_ROOT.glob("captions_task*.jsonl"))
    else:
        paths = [CAPTIONS_ROOT / f"captions_task{task_id:04d}.jsonl" for task_id in task_ids]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing caption files: {missing[:5]}")
    if not paths:
        raise FileNotFoundError(f"No caption files found under {CAPTIONS_ROOT}")
    return paths


def split_caption_paths(caption_paths: list[Path], worker_count: int) -> list[list[Path]]:
    worker_count = max(1, min(worker_count, len(caption_paths)))
    buckets = [[] for _ in range(worker_count)]
    bucket_sizes = [0] * worker_count

    for path in sorted(caption_paths, key=lambda item: item.stat().st_size, reverse=True):
        worker_index = min(range(worker_count), key=bucket_sizes.__getitem__)
        buckets[worker_index].append(path)
        bucket_sizes[worker_index] += path.stat().st_size

    for bucket in buckets:
        bucket.sort()

    return buckets


def normalize_caption(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").strip()


def iter_caption_records(
    caption_paths: list[Path],
    limit_per_task: int | None,
    limit_total: int | None,
) -> Iterator[CaptionRecord]:
    seen_total = 0
    for caption_path in caption_paths:
        seen_in_task = 0
        with caption_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                sample_id = payload.get("sample_id")
                caption = normalize_caption(payload.get("caption"))
                metadata = payload.get("metadata") or {}
                source_tar = metadata.get("source_tar")
                if not sample_id or not caption or not source_tar:
                    continue
                yield CaptionRecord(
                    sample_id=str(sample_id),
                    caption=caption,
                    source_tar=str(source_tar),
                )
                seen_in_task += 1
                seen_total += 1
                if limit_per_task is not None and seen_in_task >= limit_per_task:
                    break
                if limit_total is not None and seen_total >= limit_total:
                    return


def group_captions_by_source_tar(
    records: Iterable[CaptionRecord],
) -> Iterator[tuple[str, list[CaptionRecord]]]:
    current_source_tar: str | None = None
    current_records: list[CaptionRecord] = []

    for record in records:
        if current_source_tar is None or record.source_tar == current_source_tar:
            current_source_tar = record.source_tar
            current_records.append(record)
            continue

        yield current_source_tar, current_records
        current_source_tar = record.source_tar
        current_records = [record]

    if current_source_tar is not None:
        yield current_source_tar, current_records


def extract_image(sample: dict) -> tuple[bytes, str]:
    for key, extension in IMAGE_KEYS_TO_EXTENSIONS.items():
        image_bytes = sample.get(key)
        if image_bytes is not None:
            return bytes(image_bytes), extension

    available_keys = ", ".join(sorted(sample))
    raise KeyError(f"No supported image payload found in sample keys: {available_keys}")


def build_sample_id(source_tar_path: Path, source_key: str) -> str:
    partition_name = source_tar_path.parent.parent.name
    shard_name = source_tar_path.stem
    return f"{partition_name}__{shard_name}__{source_key}"


def iter_matched_samples(
    source_tar: str,
    records: list[CaptionRecord],
) -> Iterator[tuple[CaptionRecord, dict, Path]]:
    source_tar_path = Path(source_tar)
    wanted = {record.sample_id: record for record in records}
    matched_ids: set[str] = set()
    dataset = wds.WebDataset(str(source_tar_path), shardshuffle=False, empty_check=False)

    for sample in dataset:
        source_key = sample["__key__"]
        sample_id = build_sample_id(source_tar_path, source_key)
        record = wanted.get(sample_id)
        if record is None:
            continue
        yield record, sample, source_tar_path
        matched_ids.add(sample_id)
        if len(matched_ids) == len(records):
            break

    if len(matched_ids) != len(records):
        raise RuntimeError(
            f"Missing {len(records) - len(matched_ids)} samples in {source_tar_path}; "
            f"first missing ids: {sorted(set(wanted) - matched_ids)[:5]}"
        )


def build_metadata(record: CaptionRecord, sample: dict, source_tar_path: Path) -> dict:
    source_metadata = json.loads(bytes(sample.get("json", b"{}")).decode("utf-8"))
    source_key = sample["__key__"]
    metadata = {key: value for key, value in source_metadata.items() if key not in TEXT_METADATA_KEYS}
    metadata.update(
        {
            "sample_id": record.sample_id,
            "source_key": source_key,
            "partition": source_tar_path.parent.parent.name,
            "source_shard": source_tar_path.name,
            "source_tar": str(source_tar_path),
        }
    )
    return metadata


def build_sample_payload(
    record: CaptionRecord,
    sample: dict,
    source_tar_path: Path,
) -> dict[str, bytes]:
    image_bytes, image_ext = extract_image(sample)
    metadata = build_metadata(record, sample, source_tar_path)
    return {
        "__key__": record.sample_id,
        image_ext: image_bytes,
        "txt": record.caption.encode("utf-8"),
        "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
    }


def worker_output_pattern(output_dir: Path, worker_id: int, multi_worker: bool) -> str:
    if not multi_worker:
        return str(output_dir / "part-%06d.tar")
    return str(output_dir / f"worker{worker_id:03d}-part-%06d.tar")


def run_worker(
    worker_id: int,
    caption_paths: list[Path],
    output_dir: Path,
    limit_per_task: int | None,
    limit_total: int | None,
    shard_maxcount: int,
    shard_maxsize: int,
    multi_worker: bool,
) -> tuple[int, int, int, str]:
    output_pattern = worker_output_pattern(output_dir, worker_id, multi_worker)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    source_tar_count = 0
    with wds.ShardWriter(
        output_pattern,
        maxcount=shard_maxcount,
        maxsize=shard_maxsize,
    ) as sink:
        records = iter_caption_records(caption_paths, limit_per_task, limit_total)
        for source_tar, record_group in group_captions_by_source_tar(records):
            source_tar_count += 1
            for record, sample, source_tar_path in iter_matched_samples(source_tar, record_group):
                sink.write(build_sample_payload(record, sample, source_tar_path))
                written += 1
                if written % PROGRESS_EVERY == 0:
                    print(
                        f"[worker {worker_id}] wrote {written} samples across {source_tar_count} source shards",
                        flush=True,
                    )

    return worker_id, written, source_tar_count, output_pattern


def main() -> None:
    args = parse_args()
    task_ids = parse_task_ids(args.task_ids)
    caption_paths = list_caption_paths(task_ids)
    output_dir = resolve_output_dir(args.output_root, args.subdir)

    if args.limit_total is not None and args.workers != 1:
        raise ValueError("--limit-total currently requires --workers=1 for deterministic sampling")

    worker_caption_paths = split_caption_paths(caption_paths, args.workers)

    print(f"Using captions from {CAPTIONS_ROOT}", flush=True)
    print(f"Selected {len(caption_paths)} caption files", flush=True)
    print(
        f"Writing WebDataset to {output_dir} with {len(worker_caption_paths)} workers",
        flush=True,
    )

    total_written = 0
    total_source_tars = 0
    with ProcessPoolExecutor(max_workers=len(worker_caption_paths)) as executor:
        futures = [
            executor.submit(
                run_worker,
                worker_id,
                worker_paths,
                output_dir,
                args.limit_per_task,
                args.limit_total,
                args.shard_maxcount,
                args.shard_maxsize,
                len(worker_caption_paths) > 1,
            )
            for worker_id, worker_paths in enumerate(worker_caption_paths)
        ]
        for future in as_completed(futures):
            worker_id, written, source_tar_count, output_pattern = future.result()
            total_written += written
            total_source_tars += source_tar_count
            print(
                f"[worker {worker_id}] completed {written} samples "
                f"across {source_tar_count} source shards -> {output_pattern}",
                flush=True,
            )

    summary = {
        "caption_files": len(caption_paths),
        "limit_per_task": args.limit_per_task,
        "limit_total": args.limit_total,
        "output_dir": str(output_dir),
        "selected_task_ids": task_ids,
        "shard_maxcount": args.shard_maxcount,
        "shard_maxsize": args.shard_maxsize,
        "source_tars_processed": total_source_tars,
        "workers": len(worker_caption_paths),
        "written": total_written,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
