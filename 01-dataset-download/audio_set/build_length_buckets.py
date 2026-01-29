#!/usr/bin/env python
"""Build length buckets for AudioSet using parallel processing.

Scans cached AudioSet Arrow files to compute audio durations (via FLAC header
parsing with soundfile fallback), assigns each sample to a length bucket, and
writes per-sample metadata including the global HF dataset row index for fast
downstream selection via ds.select().

Usage:
    # Full dataset with 128 workers (default output: ./length_buckets/)
    python build_length_buckets.py --num-workers 128

    # Subset of 10k samples for testing
    python build_length_buckets.py --max-examples 10000 --num-workers 32

    # Custom output directory and split
    python build_length_buckets.py --split eval --out-dir /path/to/output

Outputs (in --out-dir):
    audioset_{split}_buckets.tsv       - video_id, global_idx, bucket, length
    audioset_{split}_utt2len.tsv       - video_id, length
    audioset_{split}_bucket_counts.tsv - bucket, count
"""

import argparse
import io
import struct
from collections import Counter
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time

import pyarrow.ipc as ipc
import soundfile as sf
from tqdm import tqdm

_FLAC_MAGIC = b"fLaC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build length buckets for AudioSet (parallel version)."
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(
            "/capstor/store/cscs/swissai/infra01/audio-datasets/audioset_cache/"
            "agkphysics___audio_set/full/0.0.0/"
            "0c609e8302cf139307f639c57652032af0a88041"
        ),
    )
    parser.add_argument("--split", type=str, default="unbal_train")
    parser.add_argument("--target-sr", type=int, default=24000)
    parser.add_argument("--bucket-size-samples", type=int, default=1000)
    parser.add_argument("--max-examples", type=int, default=0, help="0 = no limit")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "length_buckets")
    parser.add_argument("--num-workers", type=int, default=0, help="0 = use all CPUs")
    return parser.parse_args()


def find_arrow_files(cache_root: Path, split: str) -> list[Path]:
    pattern = f"audio_set-{split}-*.arrow"
    files = sorted(cache_root.glob(pattern))
    if files:
        return files
    return sorted(cache_root.rglob(pattern))


def _get_audio_length(audio_bytes: bytes, target_sr: int) -> int | None:
    """Get audio length in target_sr samples.

    Fast path: parse FLAC STREAMINFO header directly (bytes 18-25).
    Fallback: use soundfile for non-FLAC or malformed headers.
    """
    # FLAC fast path: 4B magic + 4B block header + 10B STREAMINFO fields
    # then 8 bytes at offset 18: sample_rate(20) | channels(3) | bps(5) | total_samples(36)
    if len(audio_bytes) >= 26 and audio_bytes[:4] == _FLAC_MAGIC:
        sr_and_rest = struct.unpack(">Q", audio_bytes[18:26])[0]
        sample_rate = (sr_and_rest >> 44) & 0xFFFFF
        total_samples = sr_and_rest & 0xFFFFFFFFF
        if sample_rate > 0 and total_samples > 0:
            return int(round(total_samples * (target_sr / sample_rate)))

    # Fallback to soundfile
    try:
        info = sf.info(io.BytesIO(audio_bytes))
        if info.frames is not None and info.samplerate:
            return int(round(info.frames * (target_sr / info.samplerate)))
        elif info.duration is not None:
            return int(round(info.duration * target_sr))
    except Exception:
        pass
    return None


def process_arrow_file(args: tuple) -> tuple[list, Counter, int, int, int]:
    """Process a single Arrow file.

    Args: tuple of (arrow_path, target_sr, bucket_size).
    Returns: (results, bucket_counts, processed, errors, total_rows).
        results contains (vid, local_idx, bucket, length).
    """
    arrow_path, target_sr, bucket_size = args
    results = []
    bucket_counts = Counter()
    processed = 0
    errors = 0
    local_idx = 0

    try:
        with arrow_path.open("rb") as f:
            reader = ipc.open_stream(f)
            for batch in reader:
                video_ids = batch.column("video_id")
                audio_col = batch.column("audio")
                for row in range(batch.num_rows):
                    vid = video_ids[row].as_py()
                    audio_bytes = audio_col[row]["bytes"].as_py()

                    if audio_bytes is not None:
                        length = _get_audio_length(audio_bytes, target_sr)
                        if length is not None:
                            bucket = (length // bucket_size) * bucket_size
                            results.append((vid, local_idx, bucket, length))
                            bucket_counts[bucket] += 1
                            processed += 1
                        else:
                            errors += 1
                    else:
                        errors += 1
                    local_idx += 1
    except Exception as e:
        print(f"Error processing {arrow_path}: {e}")

    return results, bucket_counts, processed, errors, local_idx


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    arrow_files = find_arrow_files(args.cache_root, args.split)
    if not arrow_files:
        print(f"No Arrow shards found under: {args.cache_root}")
        return 2

    print(f"Found {len(arrow_files)} Arrow files")

    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    print(f"Using {num_workers} workers")

    start_time = time.time()

    # Single pass: process in parallel, compute global offsets from ordered results
    worker_args = [
        (f, args.target_sr, args.bucket_size_samples) for f in arrow_files
    ]

    all_results = []
    total_counts = Counter()
    total_processed = 0
    total_errors = 0
    cumulative_offset = 0

    with Pool(num_workers) as pool:
        pbar = tqdm(
            pool.imap(process_arrow_file, worker_args),
            total=len(arrow_files),
            desc="Processing",
            unit="files",
            mininterval=30,
        )
        for results, counts, processed, errors, file_rows in pbar:
            for vid, local_idx, bucket, length in results:
                all_results.append((vid, cumulative_offset + local_idx, bucket, length))
            cumulative_offset += file_rows
            total_counts.update(counts)
            total_processed += processed
            total_errors += errors
            pbar.set_postfix(samples=f"{total_processed:,}", errors=total_errors)

            if args.max_examples > 0 and total_processed >= args.max_examples:
                pool.terminate()
                break

    elapsed = time.time() - start_time

    print(f"Total rows: {cumulative_offset:,}")

    # Write outputs
    utt2len_path = args.out_dir / f"audioset_{args.split}_utt2len.tsv"
    buckets_path = args.out_dir / f"audioset_{args.split}_buckets.tsv"
    counts_path = args.out_dir / f"audioset_{args.split}_bucket_counts.tsv"

    with utt2len_path.open("w") as f:
        for vid, global_idx, bucket, length in all_results:
            f.write(f"{vid}\t{length}\n")

    with buckets_path.open("w") as f:
        for vid, global_idx, bucket, length in all_results:
            f.write(f"{vid}\t{global_idx}\t{bucket}\t{length}\n")

    with counts_path.open("w") as f:
        for bucket, count in sorted(total_counts.items()):
            f.write(f"{bucket}\t{count}\n")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Processed: {total_processed:,} | Errors: {total_errors} | Rate: {total_processed/elapsed:.0f}/s")
    print(f"Output: {args.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
