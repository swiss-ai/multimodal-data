#!/usr/bin/env python3
"""Segment WavePulse radio MP3s using transcript timestamps and write parquet.

Uses multiprocessing with bounded memory — each worker processes one recording
and returns segments. Main thread writes shards incrementally.

Requires FFMPEG_BIN env var or ffmpeg at the default capstor path.
Set LD_LIBRARY_PATH for ffmpeg shared libs.

Usage:
    export FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
    export PATH=${FFMPEG_ROOT}/bin:${PATH}
    export LD_LIBRARY_PATH=${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}

    python segment_and_convert.py \
        --recordings-dir /capstor/.../hf___nyu-dice-lab___wavepulse-radio-raw-transcripts/recordings \
        --transcripts-dir /capstor/.../hf___nyu-dice-lab___wavepulse-radio-raw-transcripts/data \
        --output-dir /capstor/.../hf___nyu-dice-lab___wavepulse-radio-raw-transcripts/segmented_parquets \
        --num-workers 32
"""

import argparse
import glob
import io
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import subprocess

import numpy as np
import pandas as pd
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def process_one_recording(args):
    """Process one MP3 → list of segment dicts (runs in worker process)."""
    mp3_path, segments_df = args
    rec_id = Path(mp3_path).stem
    results = []

    ffmpeg_bin = os.environ.get("FFMPEG_BIN",
        "/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64/bin/ffmpeg")
    sr = 16000
    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
             "-i", mp3_path, "-vn", "-sn", "-dn",
             "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1",
             "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120,
        )
        if proc.returncode != 0 or not proc.stdout:
            return results
        audio = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return results

    for _, row in segments_df.iterrows():
        start_sample = int(row["start_time"] * sr)
        end_sample = int(row["end_time"] * sr)
        if start_sample >= len(audio) or end_sample <= start_sample:
            continue
        end_sample = min(end_sample, len(audio))

        buf = io.BytesIO()
        sf.write(buf, audio[start_sample:end_sample], sr, format="WAV", subtype="PCM_16")

        results.append({
            "id": f"{rec_id}_seg{row['segment_index']}",
            "audio": {"bytes": buf.getvalue(), "sampling_rate": sr},
            "text": row["text"],
            "duration": row["end_time"] - row["start_time"],
            "speaker": row.get("speaker", "UNKNOWN"),
            "station": row.get("station", ""),
            "state": row.get("state", ""),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings-dir", type=str, required=True)
    parser.add_argument("--transcripts-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=5000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mp3_files = sorted(Path(args.recordings_dir).glob("*.mp3"))
    recording_ids = {f.stem for f in mp3_files}
    logger.info(f"Found {len(mp3_files)} MP3 recordings")

    # Load matching transcripts
    logger.info("Loading transcripts...")
    pqs = sorted(glob.glob(f"{args.transcripts_dir}/**/*.parquet", recursive=True))
    grouped = {}
    for pq in pqs:
        df = pd.read_parquet(pq)
        for tid, grp in df[df["transcript_id"].isin(recording_ids)].groupby("transcript_id"):
            grouped.setdefault(tid, []).append(grp)
    for tid in grouped:
        grouped[tid] = pd.concat(grouped[tid]).sort_values("segment_index")
    logger.info(f"Matched {len(grouped)} recordings")

    # Build work items
    work_items = []
    for mp3_path in mp3_files:
        rec_id = mp3_path.stem
        if rec_id in grouped:
            work_items.append((str(mp3_path), grouped[rec_id]))

    # Process with bounded parallelism via imap_unordered
    # Each call returns one recording's segments, main thread flushes shards
    buffer = []
    shard_idx = 0
    total_segments = 0
    total_hours = 0.0

    with Pool(processes=args.num_workers) as pool:
        for i, segments in enumerate(pool.imap_unordered(process_one_recording, work_items, chunksize=1)):
            buffer.extend(segments)
            total_segments += len(segments)
            total_hours += sum(s["duration"] for s in segments) / 3600

            # Flush full shards
            while len(buffer) >= args.shard_size:
                shard = buffer[:args.shard_size]
                buffer = buffer[args.shard_size:]
                out_path = Path(args.output_dir) / f"train-{shard_idx:05d}.parquet"
                pd.DataFrame(shard).to_parquet(out_path)
                shard_idx += 1

            if (i + 1) % 50 == 0:
                logger.info(f"{i+1}/{len(work_items)} recordings, {total_segments} segments, {total_hours:.1f}h")

    # Flush remaining
    if buffer:
        out_path = Path(args.output_dir) / f"train-{shard_idx:05d}.parquet"
        pd.DataFrame(buffer).to_parquet(out_path)
        shard_idx += 1

    logger.info(f"Done: {total_segments} segments, {total_hours:.1f}h, {shard_idx} shards")


if __name__ == "__main__":
    main()
