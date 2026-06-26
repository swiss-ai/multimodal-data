#!/usr/bin/env python3
"""Build a NeMo-style training manifest from packed LN parquet.

Stage 2 of the LN→ASR pipeline. Reads the per-group parquet shards produced
by `ln_to_parquet.py`, applies quality filters, chunks long clips at silent
gap boundaries, and emits a single training manifest.

Each output line is one trainable utterance:

  {"audio_filepath": "ln://<group>/<shard>.parquet#row=<i>&start=<s>&end=<s>",
   "duration": <s>,
   "text": "<caption-or-chunk-text>",
   "speaker_id": "ln_<annotator_id>",
   "lang": "en"}

The audio_filepath is a logical URI — pair it with a custom Lhotse/NeMo
audio resolver that reads the parquet row, decodes audio_bytes, and slices
to [start, end]. (See ln_audio_resolver.py.)

Filters applied (drop the row):
  - speech_coverage < min_speech_coverage    (mostly silence)
  - zero_duration_ratio > max_zero_dur       (degenerate alignment)
  - total_speech < min_speech_seconds        (too short to learn from)

Chunking (split rows over max_chunk_seconds at gap boundaries):
  - target chunk length: max_chunk_seconds (default 25s)
  - cuts only at inter-word gaps > min_split_gap (default 500ms)
  - if no acceptable gap exists, leaves the clip whole and emits a warning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_PARQUET_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/localized_narratives"
)


def speech_stats(timed_caption: list[dict]) -> tuple[float, float, float, int]:
    """Return (start, end, total_speech_seconds, zero_duration_count)."""
    if not timed_caption:
        return 0.0, 0.0, 0.0, 0
    start = min(t["start_time"] for t in timed_caption)
    end = max(t["end_time"] for t in timed_caption)
    total_speech = sum(max(0.0, t["end_time"] - t["start_time"]) for t in timed_caption)
    n_zero = sum(1 for t in timed_caption if t["end_time"] - t["start_time"] < 1e-6)
    return start, end, total_speech, n_zero


def gap_split_points(timed_caption: list[dict], min_gap: float) -> list[int]:
    """Indices i such that timed_caption[i].start - timed_caption[i-1].end > min_gap.

    A split BEFORE index i means: chunk_a ends at tc[i-1].end_time,
    chunk_b starts at tc[i].start_time.
    """
    pts = []
    for i in range(1, len(timed_caption)):
        gap = timed_caption[i]["start_time"] - timed_caption[i - 1]["end_time"]
        if gap > min_gap:
            pts.append(i)
    return pts


def chunk_record(
    timed_caption: list[dict],
    *,
    max_chunk_s: float,
    min_split_gap: float,
) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) chunks over timed_caption.

    Greedy: walk through the records' words; whenever the running chunk would
    exceed max_chunk_s AND we've passed at least one valid split-point gap,
    emit a chunk ending at the most recent split-point. If no split-point
    exists in a too-long span, fall back to splitting at the longest gap in
    that span (best effort) — never mid-word.
    """
    if not timed_caption:
        return []
    chunks: list[tuple[int, int]] = []
    n = len(timed_caption)
    chunk_start_idx = 0
    chunk_start_time = timed_caption[0]["start_time"]
    last_valid_split = 0  # index where we could have split (gap > min_split_gap)

    for i in range(1, n):
        prev_end = timed_caption[i - 1]["end_time"]
        cur_start = timed_caption[i]["start_time"]
        gap = cur_start - prev_end
        if gap > min_split_gap:
            last_valid_split = i

        running_dur = prev_end - chunk_start_time
        if running_dur > max_chunk_s:
            # Need to split.
            if last_valid_split > chunk_start_idx:
                # Split at the most recent acceptable gap.
                chunks.append((chunk_start_idx, last_valid_split))
                chunk_start_idx = last_valid_split
                chunk_start_time = timed_caption[chunk_start_idx]["start_time"]
                last_valid_split = chunk_start_idx
            else:
                # No acceptable gap in this overshoot → fall back to longest gap.
                best_i, best_gap = chunk_start_idx + 1, -1.0
                for j in range(chunk_start_idx + 1, i):
                    g = timed_caption[j]["start_time"] - timed_caption[j - 1]["end_time"]
                    if g > best_gap:
                        best_gap = g
                        best_i = j
                if best_i > chunk_start_idx:
                    chunks.append((chunk_start_idx, best_i))
                    chunk_start_idx = best_i
                    chunk_start_time = timed_caption[chunk_start_idx]["start_time"]
                    last_valid_split = chunk_start_idx

    # Final chunk
    if chunk_start_idx < n:
        chunks.append((chunk_start_idx, n))
    return chunks


def chunk_to_entry(
    parquet_uri: str,
    row_idx: int,
    timed_caption: list[dict],
    chunk: tuple[int, int],
    annotator_id: int,
    image_id: str,
) -> dict:
    a, b = chunk
    start = timed_caption[a]["start_time"]
    end = timed_caption[b - 1]["end_time"]
    text = " ".join(t["utterance"] for t in timed_caption[a:b] if t.get("utterance"))
    return {
        "audio_filepath": f"{parquet_uri}#row={row_idx}&start={start:.3f}&end={end:.3f}",
        "duration": round(end - start, 3),
        "text": text,
        "speaker_id": f"ln_{annotator_id}",
        "lang": "en",
        "image_id": image_id,
    }


def process_parquet(
    parquet_path: Path,
    *,
    parquet_root: Path,
    out_fp,
    args,
    stats: dict,
) -> None:
    """Stream rows from one parquet, apply filters/chunking, write manifest lines."""
    rel = parquet_path.relative_to(parquet_root)
    parquet_uri = f"ln://{rel.as_posix()}"

    pf = pq.ParquetFile(str(parquet_path))
    # Columns: skip audio_bytes (we don't need it for manifest, just metadata).
    cols = ["image_id", "annotator_id", "caption", "timed_caption", "voice_recording_path"]
    row_idx_global = 0
    for batch in pf.iter_batches(batch_size=4096, columns=cols):
        recs = batch.to_pylist()
        for rec in recs:
            stats["seen"] += 1
            tc = rec["timed_caption"] or []
            if not tc:
                stats["filt_empty"] += 1
                row_idx_global += 1
                continue
            start, end, total_speech, n_zero = speech_stats(tc)
            clip_dur = max(end - start, 1e-6)
            speech_cov = total_speech / clip_dur
            zero_ratio = n_zero / len(tc)

            if total_speech < args.min_speech_seconds:
                stats["filt_too_short"] += 1
                row_idx_global += 1
                continue
            if speech_cov < args.min_speech_coverage:
                stats["filt_low_coverage"] += 1
                row_idx_global += 1
                continue
            if zero_ratio > args.max_zero_dur_ratio:
                stats["filt_zero_dur"] += 1
                row_idx_global += 1
                continue

            chunks = chunk_record(
                tc,
                max_chunk_s=args.max_chunk_seconds,
                min_split_gap=args.min_split_gap,
            )
            if not chunks:
                stats["filt_no_chunks"] += 1
                row_idx_global += 1
                continue

            for ch in chunks:
                ch_dur = tc[ch[1] - 1]["end_time"] - tc[ch[0]]["start_time"]
                if ch_dur < args.min_speech_seconds:
                    stats["chunk_dropped_short"] += 1
                    continue
                entry = chunk_to_entry(
                    parquet_uri,
                    row_idx_global,
                    tc,
                    ch,
                    rec["annotator_id"],
                    rec["image_id"],
                )
                out_fp.write(json.dumps(entry, ensure_ascii=False))
                out_fp.write("\n")
                stats["emitted"] += 1
                stats["emitted_seconds"] += ch_dur
            row_idx_global += 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT,
                   help="Root containing <group>/<group>-NNNNN.parquet")
    p.add_argument("--output", type=Path, required=True,
                   help="Output manifest .jsonl path")
    p.add_argument("--groups", default="all",
                   help="Comma-separated group names, or 'all'")

    p.add_argument("--max-chunk-seconds", type=float, default=25.0)
    p.add_argument("--min-split-gap", type=float, default=0.5,
                   help="Only split clips at inter-word gaps > this (seconds)")
    p.add_argument("--min-speech-seconds", type=float, default=1.0,
                   help="Drop clips/chunks with less total speech than this")
    p.add_argument("--min-speech-coverage", type=float, default=0.6,
                   help="Drop clips where total_speech / clip_duration < this")
    p.add_argument("--max-zero-dur-ratio", type=float, default=0.20,
                   help="Drop clips where >this fraction of timed entries have zero duration")
    args = p.parse_args()

    if args.groups == "all":
        group_dirs = sorted([d for d in args.parquet_root.iterdir() if d.is_dir()])
    else:
        group_dirs = [args.parquet_root / g.strip() for g in args.groups.split(",")]
        for g in group_dirs:
            if not g.is_dir():
                print(f"ERROR: group dir not found: {g}", file=sys.stderr)
                return 2

    stats = {
        "seen": 0, "emitted": 0, "emitted_seconds": 0.0,
        "filt_empty": 0, "filt_too_short": 0, "filt_low_coverage": 0,
        "filt_zero_dur": 0, "filt_no_chunks": 0, "chunk_dropped_short": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    print(f"Building manifest from {len(group_dirs)} groups → {args.output}")
    print(f"Filters: min_speech={args.min_speech_seconds}s, "
          f"min_coverage={args.min_speech_coverage}, "
          f"max_zero_dur={args.max_zero_dur_ratio}")
    print(f"Chunking: max={args.max_chunk_seconds}s, "
          f"min_split_gap={args.min_split_gap}s")
    sys.stdout.flush()

    with args.output.open("w") as out_fp:
        for gd in group_dirs:
            shards = sorted(gd.glob("*.parquet"))
            if not shards:
                print(f"  WARN: no parquet in {gd}", file=sys.stderr)
                continue
            t_g = time.monotonic()
            seen_before = stats["seen"]
            emitted_before = stats["emitted"]
            for sh in shards:
                process_parquet(sh, parquet_root=args.parquet_root,
                                out_fp=out_fp, args=args, stats=stats)
            dt = time.monotonic() - t_g
            d_seen = stats["seen"] - seen_before
            d_emit = stats["emitted"] - emitted_before
            print(f"  {gd.name:<24} {d_seen:>7d} rows → {d_emit:>7d} chunks  "
                  f"({dt:5.1f}s, {len(shards)} shards)", flush=True)

    elapsed = time.monotonic() - t0
    print(f"\nManifest written: {args.output}")
    print(f"Total rows seen:        {stats['seen']:>10d}")
    print(f"Total chunks emitted:   {stats['emitted']:>10d}")
    print(f"Total emitted hours:    {stats['emitted_seconds']/3600:>10.1f}")
    print(f"Filtered (empty):       {stats['filt_empty']:>10d}")
    print(f"Filtered (too short):   {stats['filt_too_short']:>10d}")
    print(f"Filtered (low cover):   {stats['filt_low_coverage']:>10d}")
    print(f"Filtered (zero-dur):    {stats['filt_zero_dur']:>10d}")
    print(f"Filtered (no chunks):   {stats['filt_no_chunks']:>10d}")
    print(f"Chunks dropped short:   {stats['chunk_dropped_short']:>10d}")
    print(f"Elapsed:                {elapsed:>10.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
