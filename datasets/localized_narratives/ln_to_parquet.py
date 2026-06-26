#!/usr/bin/env python3
"""Pack Localized Narratives raw tree into per-group parquet.

Input:  raw/localized_narratives/{annotations/*.jsonl, voice-recordings/<group>/*.ogg}
Output: processed/localized_narratives/<group>[-NNNNN].parquet

Each row carries the full annotation (caption, word-level timed_caption,
mouse traces) plus the OGG bytes inlined as a binary column. This collapses
~870k voice-recording inodes into a few dozen parquet files while keeping
the data lossless and queryable.

Run one --group per node (SLURM array). Splits with >50k records can be
sharded into N-row parquet files via --shard-rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Annotation glob per group. Sharded JSONLs (open_images_train,
# coco_train) are matched via the trailing wildcard.
GROUPS: dict[str, str] = {
    "open_images_train":      "open_images_train_v6_localized_narratives*.jsonl",
    "open_images_validation": "open_images_validation_localized_narratives.jsonl",
    "open_images_test":       "open_images_test_localized_narratives.jsonl",
    "coco_train":             "coco_train_localized_narratives*.jsonl",
    "coco_val":               "coco_val_localized_narratives.jsonl",
    "flickr30k_train":        "flickr30k_train_localized_narratives.jsonl",
    "flickr30k_val":          "flickr30k_val_localized_narratives.jsonl",
    "flickr30k_test":         "flickr30k_test_localized_narratives.jsonl",
    "ade20k_train":           "ade20k_train_localized_narratives.jsonl",
    "ade20k_validation":      "ade20k_validation_localized_narratives.jsonl",
}

SCHEMA = pa.schema([
    ("dataset_id",           pa.string()),
    ("image_id",             pa.string()),
    ("annotator_id",         pa.int32()),
    ("caption",              pa.string()),
    ("timed_caption",        pa.list_(pa.struct([
        ("utterance",   pa.string()),
        ("start_time",  pa.float32()),
        ("end_time",    pa.float32()),
    ]))),
    ("traces",               pa.list_(pa.list_(pa.struct([
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("t", pa.float32()),
    ])))),
    ("voice_recording_path", pa.string()),
    ("audio_bytes",          pa.binary()),
])


def iter_records(annotation_paths: list[Path]):
    for path in annotation_paths:
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def read_bytes(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.read()


class ShardingWriter:
    """ParquetWriter that rotates files every shard_rows records.

    shard_rows=None → single un-suffixed file `<group>.parquet`.
    """

    def __init__(self, output_dir: Path, group: str, shard_rows: int | None):
        self.output_dir = output_dir
        self.group = group
        self.shard_rows = shard_rows
        self.shard_idx = 0
        self.rows_in_shard = 0
        self.writer: pq.ParquetWriter | None = None

    def _open_next(self) -> None:
        name = (
            f"{self.group}-{self.shard_idx:05d}.parquet"
            if self.shard_rows else f"{self.group}.parquet"
        )
        path = self.output_dir / name
        self.writer = pq.ParquetWriter(
            str(path), SCHEMA, compression="zstd", compression_level=3,
        )
        self.rows_in_shard = 0

    def write_batch(self, batch: pa.RecordBatch) -> None:
        if self.writer is None:
            self._open_next()
        if self.shard_rows:
            remaining = self.shard_rows - self.rows_in_shard
            if batch.num_rows > remaining:
                head = batch.slice(0, remaining)
                tail = batch.slice(remaining, batch.num_rows - remaining)
                self.writer.write_batch(head)
                self.rows_in_shard += head.num_rows
                self.writer.close()
                self.shard_idx += 1
                self._open_next()
                self.write_batch(tail)
                return
        self.writer.write_batch(batch)
        self.rows_in_shard += batch.num_rows

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def build_batch(rows: list[tuple[dict, bytes]]) -> pa.RecordBatch:
    cols: dict[str, list] = {name: [] for name in SCHEMA.names}
    for rec, audio in rows:
        cols["dataset_id"].append(rec["dataset_id"])
        cols["image_id"].append(rec["image_id"])
        cols["annotator_id"].append(int(rec["annotator_id"]))
        cols["caption"].append(rec["caption"])
        cols["timed_caption"].append(rec["timed_caption"])
        cols["traces"].append(rec["traces"])
        cols["voice_recording_path"].append(rec["voice_recording"])
        cols["audio_bytes"].append(audio)
    arrays = [pa.array(cols[f.name], type=f.type) for f in SCHEMA]
    return pa.RecordBatch.from_arrays(arrays, schema=SCHEMA)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-root", required=True, type=Path,
                   help="raw/localized_narratives root (contains annotations/, voice-recordings/)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="processed/localized_narratives/<group>/ (created if missing)")
    p.add_argument("--group", required=True, choices=sorted(GROUPS.keys()))
    p.add_argument("--shard-rows", type=int, default=0,
                   help="Split output into N-row shards (0 = single parquet)")
    p.add_argument("--audio-workers", type=int, default=256,
                   help="Threaded OGG readers. Tune to match node CPU count "
                        "(default 256 ≈ 288-core node minus headroom for writer)")
    p.add_argument("--batch-rows", type=int, default=4000,
                   help="Flush every N records (larger = fewer Arrow array "
                        "constructions, more steady throughput)")
    p.add_argument("--in-flight-mult", type=int, default=4,
                   help="Max in-flight reads = audio_workers × this. "
                        "Higher = more pipelining but more memory")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ann_root = args.source_root / "annotations"
    audio_root = args.source_root / "voice-recordings"

    ann_files = sorted(ann_root.glob(GROUPS[args.group]))
    if not ann_files:
        print(f"ERROR: no annotation files matched {GROUPS[args.group]} in {ann_root}",
              file=sys.stderr)
        return 2

    print(f"Group:            {args.group}")
    print(f"Annotation files: {len(ann_files)}")
    for f in ann_files:
        print(f"  {f.name}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Shard rows:       {args.shard_rows or 'unsharded'}")
    print(f"Audio workers:    {args.audio_workers}")
    print(f"Batch rows:       {args.batch_rows}")
    sys.stdout.flush()

    writer = ShardingWriter(args.output_dir, args.group, args.shard_rows or None)
    pool = ThreadPoolExecutor(max_workers=args.audio_workers)
    max_in_flight = args.audio_workers * args.in_flight_mult

    t0 = time.monotonic()
    n_ok = 0
    n_missing = 0
    last_log = 0
    in_flight: deque = deque()           # ordered (rec, future) waiting on IO
    ready_batch: list[tuple[dict, bytes]] = []  # results ready for writer

    def drain_one() -> None:
        """Pop the oldest in-flight read, await it, push to ready_batch."""
        nonlocal n_missing
        rec, fut = in_flight.popleft()
        try:
            ready_batch.append((rec, fut.result()))
        except FileNotFoundError:
            n_missing += 1

    def flush_ready() -> None:
        """Build Arrow batch from ready rows and write it out."""
        nonlocal n_ok, last_log
        if not ready_batch:
            return
        writer.write_batch(build_batch(ready_batch))
        n_ok += len(ready_batch)
        ready_batch.clear()
        # Throttled progress: log at most every 5 batches
        if n_ok - last_log >= args.batch_rows * 5:
            elapsed = time.monotonic() - t0
            print(f"  wrote {n_ok:>8d} rows  miss={n_missing}  "
                  f"{n_ok/max(elapsed,1e-6):7.1f} rec/s", flush=True)
            last_log = n_ok

    try:
        # Rolling-window submission: keep up to max_in_flight reads queued
        # so workers never drain between batches. When we hit the cap, drain
        # the oldest one (preserves record order) and accumulate ready rows
        # until we have a full batch to write.
        for rec in iter_records(ann_files):
            audio_path = audio_root / rec["voice_recording"]
            in_flight.append((rec, pool.submit(read_bytes, audio_path)))
            while len(in_flight) >= max_in_flight:
                drain_one()
                if len(ready_batch) >= args.batch_rows:
                    flush_ready()
        # Drain remaining in-flight reads
        while in_flight:
            drain_one()
            if len(ready_batch) >= args.batch_rows:
                flush_ready()
        flush_ready()
    finally:
        pool.shutdown(wait=True)
        writer.close()

    elapsed = time.monotonic() - t0
    print(f"done. {n_ok} rows ({n_missing} missing audio) in "
          f"{elapsed:.1f}s ({n_ok/max(elapsed,1e-6):.1f} rec/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
