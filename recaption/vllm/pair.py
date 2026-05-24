#!/usr/bin/env python3

import glob
import hashlib
import json
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/hf___common-canvas___commoncatalog-cc-by")
CAPTION_GLOB = "outputs/commoncatalog/captions_task*.jsonl"
OUTPUT_PATTERN = "/path/to/data/vision-datasets/hf___common-canvas___commoncatalog-cc-by___recap/part-%06d.tar"
NUM_WORKERS = 64
SHARD_MAXCOUNT = 10_000
SHARD_MAXSIZE = 5_000_000_000
PROGRESS_EVERY = 1_000


@dataclass(frozen=True)
class CaptionRecord:
    parquet_path: str
    row_index: int
    url: str
    caption: str


def list_caption_paths() -> list[Path]:
    paths = sorted(Path(path) for path in glob.glob(CAPTION_GLOB))
    if not paths:
        raise FileNotFoundError(f"No caption files matched {CAPTION_GLOB!r}")
    return paths


def split_caption_paths(caption_paths: list[Path]) -> list[list[Path]]:
    worker_count = min(NUM_WORKERS, len(caption_paths))
    buckets = [[] for _ in range(worker_count)]
    bucket_sizes = [0] * worker_count

    for path in sorted(caption_paths, key=lambda item: item.stat().st_size, reverse=True):
        worker_index = min(range(worker_count), key=bucket_sizes.__getitem__)
        buckets[worker_index].append(path)
        bucket_sizes[worker_index] += path.stat().st_size

    for bucket in buckets:
        bucket.sort()

    return buckets


def iter_caption_records(caption_paths: list[Path]) -> Iterator[CaptionRecord]:
    for caption_path in caption_paths:
        with caption_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                yield parse_caption_record(payload)


def parse_caption_record(payload: dict) -> CaptionRecord:
    if "parquet_path" in payload and "row_index" in payload and "url" in payload:
        return CaptionRecord(
            parquet_path=payload["parquet_path"],
            row_index=int(payload["row_index"]),
            url=payload["url"],
            caption=payload["caption"],
        )

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise KeyError("metadata")

    parquet_path = metadata["path"]
    row_index = int(metadata["row_index"])
    url = metadata["url"]
    return CaptionRecord(
        parquet_path=parquet_path,
        row_index=row_index,
        url=url,
        caption=payload["caption"],
    )


def group_captions_by_parquet(
    records: Iterable[CaptionRecord],
) -> Iterator[tuple[str, list[CaptionRecord]]]:
    current_parquet: str | None = None
    current_records: list[CaptionRecord] = []

    for record in records:
        if current_parquet is None or record.parquet_path == current_parquet:
            current_parquet = record.parquet_path
            current_records.append(record)
            continue

        yield current_parquet, current_records
        current_parquet = record.parquet_path
        current_records = [record]

    if current_parquet is not None:
        yield current_parquet, current_records


def iter_matched_rows(
    parquet_path: str,
    records: list[CaptionRecord],
) -> Iterator[tuple[CaptionRecord, bytes]]:
    parquet_file = pq.ParquetFile(DATASET_ROOT / parquet_path)
    row_group_start = 0
    record_index = 0

    for row_group_index in range(parquet_file.num_row_groups):
        if record_index >= len(records):
            break

        row_group_rows = parquet_file.metadata.row_group(row_group_index).num_rows
        row_group_end = row_group_start + row_group_rows

        if records[record_index].row_index >= row_group_end:
            row_group_start = row_group_end
            continue

        table = parquet_file.read_row_group(
            row_group_index,
            columns=["jpg", "url"],
        )
        jpg_column = table["jpg"]
        url_column = table["url"]

        while record_index < len(records):
            record = records[record_index]
            if record.row_index >= row_group_end:
                break

            offset = record.row_index - row_group_start
            jpg_bytes = jpg_column[offset].as_py()
            dataset_url = url_column[offset].as_py()
            assert dataset_url == record.url, (
                f"URL mismatch for {record.parquet_path}:{record.row_index}: {dataset_url!r} != {record.url!r}"
            )

            yield record, jpg_bytes
            record_index += 1

        row_group_start = row_group_end

    if record_index != len(records):
        missing = records[record_index]
        raise RuntimeError(f"Could not locate row_index={missing.row_index} in {parquet_path}")


def make_sample_key(record: CaptionRecord) -> str:
    raw = f"{record.parquet_path}:{record.row_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def build_sample(record: CaptionRecord, jpg_bytes: bytes) -> dict[str, bytes | str]:
    metadata = {
        "url": record.url,
        "parquet_path": record.parquet_path,
        "row_index": record.row_index,
    }
    return {
        "__key__": make_sample_key(record),
        "jpg": jpg_bytes,
        "txt": record.caption.encode("utf-8"),
        "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
    }


def worker_output_pattern(worker_id: int) -> str:
    if NUM_WORKERS == 1:
        return OUTPUT_PATTERN

    output_path = Path(OUTPUT_PATTERN)
    return str(output_path.with_name(f"worker{worker_id:02d}-" + output_path.name))


def run_worker(worker_id: int, caption_paths: list[Path]) -> tuple[int, int, int, str]:
    output_pattern = worker_output_pattern(worker_id)
    output_path = Path(output_pattern)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[worker {worker_id}] using {len(caption_paths)} caption files -> {output_pattern}",
        flush=True,
    )

    written = 0
    parquet_count = 0
    with wds.ShardWriter(
        output_pattern,
        maxcount=SHARD_MAXCOUNT,
        maxsize=SHARD_MAXSIZE,
    ) as writer:
        for parquet_path, parquet_records in group_captions_by_parquet(iter_caption_records(caption_paths)):
            parquet_count += 1
            for record, jpg_bytes in iter_matched_rows(parquet_path, parquet_records):
                writer.write(build_sample(record, jpg_bytes))
                written += 1
                if PROGRESS_EVERY and written % PROGRESS_EVERY == 0:
                    print(
                        f"[worker {worker_id}] wrote {written} samples across {parquet_count} parquet files",
                        flush=True,
                    )

    return worker_id, written, parquet_count, output_pattern


def main() -> None:
    caption_paths = list_caption_paths()
    worker_caption_paths = split_caption_paths(caption_paths)

    print(f"Using captions from {CAPTION_GLOB}", flush=True)
    print(
        f"Writing WebDataset to {OUTPUT_PATTERN} with {len(worker_caption_paths)} workers",
        flush=True,
    )

    total_written = 0
    total_parquet = 0
    with ProcessPoolExecutor(max_workers=len(worker_caption_paths)) as executor:
        futures = [
            executor.submit(run_worker, worker_id, worker_paths)
            for worker_id, worker_paths in enumerate(worker_caption_paths)
        ]
        for future in as_completed(futures):
            worker_id, written, parquet_count, output_pattern = future.result()
            total_written += written
            total_parquet += parquet_count
            print(
                f"[worker {worker_id}] completed {written} samples "
                f"across {parquet_count} parquet files -> {output_pattern}",
                flush=True,
            )

    print(
        f"Completed WebDataset export: {total_written} samples across {total_parquet} parquet files",
        flush=True,
    )


if __name__ == "__main__":
    main()
