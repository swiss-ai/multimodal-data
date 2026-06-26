"""Pack CCpodcasts pre-segmented WAVs into HF-style parquet shards.

Input
-----
- ``segments.jsonl`` — one line per segment with fields:
    ``audio_filepath, duration, text, parent_audio, parent_start, parent_end, segment_idx``
- A flat directory of segment ``*.wav`` files (basenames matching
  ``audio_filepath`` in the JSONL).

Output
------
``train-NNNNN-of-MMMMM.parquet`` shards (default ~5000 segments/shard) with
HF-audio schema, suitable for ingestion via the ``parquet`` family of the
audio tokenization pipeline. Schema (all segments.jsonl fields preserved
for downstream interleave + provenance):

    audio:        {"bytes": <wav-bytes>, "path": "<basename>.wav"}
    text:         segment transcription
    duration:     float64, seconds
    source_id:    basename of parent_audio without extension (one per episode)
    segment_idx:  int, dense within source_id (used as clip_num for interleave)
    parent_audio: original parent path (provenance)
    parent_start: float64, seconds — segment offset within parent
    parent_end:   float64, seconds

The segments.jsonl ``audio_filepath`` field is *not* trusted: it points at a
stale producer-side path (``/iopsstor/.../arsaikia/long_audio/...``). This
script resolves WAVs by joining ``--audio-root`` with the JSONL basename.

Efficiency notes
----------------
- Each worker writes its own parquet shard via a streaming
  ``pyarrow.ParquetWriter`` (one batch flushed every ``--write-batch-size``
  segments). Audio bytes never cross the worker→main IPC boundary.
- Per-worker peak memory ≈ batch_size × avg_wav_bytes (~100 × 700KB ≈ 70 MB)
  plus pyarrow buffer overhead. At ``SLURM_CPUS_PER_TASK=288`` that's ~60 GB
  total, well under the 450 GB GH200 node budget.
- Segments pre-sorted by ``(parent_audio, segment_idx)`` so consecutive
  clips of one episode cluster within a shard (cheaper convert-stage reads).

Usage
-----
    python convert_to_parquet.py \\
        --segments-jsonl /capstor/.../ccpodcasts/aligned/segments.jsonl \\
        --audio-root     /capstor/.../ccpodcasts/aligned/dataset_b_segments \\
        --output-dir     /capstor/.../ccpodcasts/parquet
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# Output parquet schema — kept in sync with the audio_tokenization parquet
# runner's expectations (audio struct shape; everything else preserved as
# custom_columns).
_SCHEMA = pa.schema([
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("text", pa.string()),
    ("duration", pa.float64()),
    ("source_id", pa.string()),
    ("segment_idx", pa.int32()),
    ("parent_audio", pa.string()),
    ("parent_start", pa.float64()),
    ("parent_end", pa.float64()),
])


def _read_segments(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _source_id_from_parent(parent_audio: str) -> str:
    """Episode-stable id derived from the parent audio basename.

    The trailing portion is the producer-side hash (``...NA``), not a digit
    run, so ``input_clip_id_parser=trailing_number`` will safely split
    ``cut.id = f"{source_id}_{segment_idx}"`` into the right pair on the
    pipeline side.
    """
    return Path(parent_audio).stem


def _pack_one_shard(args: tuple[int, list[dict], str, str, int]) -> tuple[int, int, int]:
    """Worker: stream one shard's segments → parquet on disk.

    Returns ``(shard_idx, n_written, n_skipped_missing)``. No row data crosses
    the IPC boundary back to main.
    """
    shard_idx, segments, audio_root, out_path, batch_size = args
    audio_root_p = Path(audio_root)

    written = 0
    skipped = 0
    batch: list[dict] = []

    with pq.ParquetWriter(out_path, _SCHEMA, compression="snappy") as writer:
        def _flush():
            nonlocal batch
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=_SCHEMA))
                batch = []

        for seg in segments:
            wav_basename = Path(seg["audio_filepath"]).name
            wav_path = audio_root_p / wav_basename
            try:
                with open(wav_path, "rb") as f:
                    audio_bytes = f.read()
            except FileNotFoundError:
                skipped += 1
                continue
            batch.append({
                "audio": {"bytes": audio_bytes, "path": wav_basename},
                "text": seg["text"],
                "duration": float(seg["duration"]),
                "source_id": _source_id_from_parent(seg["parent_audio"]),
                "segment_idx": int(seg["segment_idx"]),
                "parent_audio": seg["parent_audio"],
                "parent_start": float(seg.get("parent_start", 0.0)),
                "parent_end": float(seg.get("parent_end", 0.0)),
            })
            written += 1
            if len(batch) >= batch_size:
                _flush()
        _flush()

    return shard_idx, written, skipped


def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments-jsonl", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-parquet", type=int, default=5000)
    parser.add_argument("--write-batch-size", type=int, default=100,
                        help="Segments per pyarrow batch flush (caps per-worker peak memory).")
    parser.add_argument(
        "--num-workers", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1),
        help="Default = SLURM_CPUS_PER_TASK if set, else os.cpu_count(). "
             "Streaming-write design keeps per-worker memory at "
             "batch_size × avg_wav (~70 MB), so saturating the node is safe.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.segments_jsonl} ...")
    segments = _read_segments(args.segments_jsonl)
    n_episodes = len({s["parent_audio"] for s in segments})
    print(f"  {len(segments):,} segments, {n_episodes:,} parent episodes")

    # Sort so consecutive segments of one episode land in the same shard.
    # Not load-bearing for interleave correctness (_detect_runs re-sorts),
    # but reduces parquet-shard fan-out per episode for the convert stage.
    segments.sort(key=lambda s: (s["parent_audio"], s["segment_idx"]))

    shards = _chunk(segments, args.samples_per_parquet)
    n_shards = len(shards)
    print(f"packing {n_shards} shards × ~{args.samples_per_parquet} segments "
          f"on {args.num_workers} workers (batch={args.write_batch_size}) ...")

    tasks = [
        (
            i,
            shard,
            str(args.audio_root),
            str(args.output_dir / f"train-{i:05d}-of-{n_shards:05d}.parquet"),
            args.write_batch_size,
        )
        for i, shard in enumerate(shards)
    ]

    total_written = 0
    total_skipped = 0
    with Pool(args.num_workers) as pool:
        for shard_idx, written, skipped in pool.imap_unordered(_pack_one_shard, tasks):
            total_written += written
            total_skipped += skipped
            print(f"  shard {shard_idx:05d}: {written} written, {skipped} skipped "
                  f"(total: {total_written:,} written, {total_skipped:,} skipped)")

    print(f"done: {total_written:,} segments packed into {n_shards} parquet files "
          f"({total_skipped:,} skipped due to missing wav)")


if __name__ == "__main__":
    main()
