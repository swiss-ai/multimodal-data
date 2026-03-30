#!/usr/bin/env python
"""Precompute audio metadata (duration, sample_rate) for People's Speech WDS.

Uses mutagen for fast MP3 header parsing, parallelized across all CPUs.
This is a one-time preprocessing step that enables fast filtering during tokenization.

Usage:
    # Full dataset with 288 workers
    python precompute_metadata.py --num-workers 288

    # Test on first 10 shards
    python precompute_metadata.py --limit-shards 10 --num-workers 32

    # Custom paths
    python precompute_metadata.py \
        --data-root /path/to/wds \
        --out-dir /path/to/output

Outputs (in --out-dir):
    peoples_speech_metadata.tsv  - shard, sample_key, sample_rate, duration, num_samples, error
    peoples_speech_summary.json  - dataset statistics
"""

import argparse
import io
import json
import tarfile
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

from mutagen.mp3 import MP3
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute audio metadata for People's Speech WDS"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/iopsstor/scratch/cscs/xyixuan/audio-datasets/unsupervised_peoples_speech_commercial_wds"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <data-root>/metadata",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel workers (0 = all CPUs)",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=0,
        help="Limit number of shards to process (0 = all)",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=24000,
        help="Target sample rate for num_samples calculation",
    )
    return parser.parse_args()


def get_mp3_metadata(data: bytes) -> tuple[int | None, float | None, int | None, str | None]:
    """Parse MP3 header to get sample_rate, duration, num_samples.

    Uses mutagen for fast header parsing (no full decode needed).
    Returns: (sample_rate, duration, num_samples, error)
    """
    try:
        mp3 = MP3(io.BytesIO(data))
        sample_rate = mp3.info.sample_rate
        duration = mp3.info.length  # seconds
        num_samples = int(duration * sample_rate) if sample_rate and duration else None
        return sample_rate, duration, num_samples, None
    except Exception as e:
        return None, None, None, str(e)[:50]


def find_tar_files(data_root: Path) -> list[Path]:
    """Find all tar files in audio/ and audio2/ subdirectories."""
    tar_files = []
    for subdir in ["audio", "audio2"]:
        subpath = data_root / subdir
        if subpath.exists():
            tar_files.extend(sorted(subpath.glob("*.tar")))
    return tar_files


def process_tar_shard(args: tuple[Path, Path]) -> tuple[str, int, int, list]:
    """Process a single tar shard and write per-shard TSV immediately.

    Args:
        args: (tar_path, shards_dir) tuple

    Returns: (shard_name, processed_count, error_count, results_for_combined)
    """
    tar_path, shards_dir = args
    results = []
    processed = 0
    errors = 0

    # Build shard identifier: audio/000001
    shard_name = f"{tar_path.parent.name}/{tar_path.stem}"
    shard_filename = shard_name.replace("/", "_") + ".tsv"
    shard_tsv_path = shards_dir / shard_filename

    try:
        with tarfile.open(tar_path, "r") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue

                # Only process MP3 files
                if not member.name.lower().endswith(".mp3"):
                    continue

                # Use full path without extension, lowercased to match WDS behavior
                # WDS lowercases extension keys, so we lowercase for consistent matching
                sample_key = str(Path(member.name).with_suffix('')).lower()

                try:
                    f = tf.extractfile(member)
                    if f is None:
                        results.append((shard_name, sample_key, None, None, None, "extract_failed"))
                        errors += 1
                        continue

                    data = f.read()
                    sample_rate, duration, num_samples, error = get_mp3_metadata(data)

                    results.append((shard_name, sample_key, sample_rate, duration, num_samples, error))
                    if error:
                        errors += 1
                    else:
                        processed += 1

                except Exception as e:
                    results.append((shard_name, sample_key, None, None, None, str(e)[:50]))
                    errors += 1

    except Exception as e:
        print(f"Error opening {tar_path}: {e}")
        results.append((shard_name, "__shard_error__", None, None, None, str(e)[:50]))
        errors += 1

    # Write per-shard TSV immediately
    with open(shard_tsv_path, "w") as f:
        f.write("sample_key\tsample_rate\tduration\tnum_samples\terror\n")
        for _, key, sr, dur, ns, err in results:
            f.write(f"{key}\t{sr or ''}\t{dur or ''}\t{ns or ''}\t{err or ''}\n")

    return shard_name, processed, errors, results


def main() -> int:
    args = parse_args()

    # Default output to <data-root>/metadata
    if args.out_dir is None:
        args.out_dir = args.data_root / "metadata"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Find tar files
    tar_files = find_tar_files(args.data_root)
    if not tar_files:
        print(f"No tar files found under: {args.data_root}")
        return 2

    if args.limit_shards > 0:
        tar_files = tar_files[:args.limit_shards]

    print(f"Found {len(tar_files)} tar shards")
    print(f"Target sample rate: {args.target_sr} Hz")

    num_workers = args.num_workers if args.num_workers > 0 else cpu_count()
    print(f"Using {num_workers} workers")

    # Create shards directory upfront so workers can write immediately
    shards_dir = args.out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    print(f"Per-shard TSVs will be written to: {shards_dir}")

    # Process in parallel - each worker writes its per-shard TSV immediately
    start_time = time.time()
    all_results = []
    total_processed = 0
    total_errors = 0

    # Pass (tar_path, shards_dir) tuples to workers
    work_items = [(tar_path, shards_dir) for tar_path in tar_files]

    with Pool(num_workers) as pool:
        pbar = tqdm(
            pool.imap(process_tar_shard, work_items),
            total=len(tar_files),
            desc="Processing shards",
            unit="shards",
        )
        for shard_name, processed, errors, results in pbar:
            all_results.extend(results)
            total_processed += processed
            total_errors += errors
            pbar.set_postfix(samples=f"{total_processed:,}", errors=total_errors)

    elapsed = time.time() - start_time

    # Write combined TSV for convenience
    tsv_path = args.out_dir / "peoples_speech_metadata.tsv"
    print(f"Writing combined {len(all_results)} entries to {tsv_path}")

    with open(tsv_path, "w") as f:
        f.write("shard\tsample_key\tsample_rate\tduration\tnum_samples\terror\n")
        for shard, key, sr, dur, ns, err in all_results:
            f.write(f"{shard}\t{key}\t{sr or ''}\t{dur or ''}\t{ns or ''}\t{err or ''}\n")

    # Compute statistics
    valid = [(sr, dur, ns) for _, _, sr, dur, ns, err in all_results if sr and dur and not err]

    if valid:
        sample_rates = [v[0] for v in valid]
        durations = [v[1] for v in valid]

        sr_counts = Counter(sample_rates)

        # Duration buckets
        dur_buckets = Counter()
        for d in durations:
            if d < 1:
                dur_buckets["<1s"] += 1
            elif d < 5:
                dur_buckets["1-5s"] += 1
            elif d < 10:
                dur_buckets["5-10s"] += 1
            elif d < 30:
                dur_buckets["10-30s"] += 1
            elif d < 60:
                dur_buckets["30-60s"] += 1
            elif d < 120:
                dur_buckets["1-2min"] += 1
            elif d < 300:
                dur_buckets["2-5min"] += 1
            else:
                dur_buckets[">5min"] += 1

        summary = {
            "data_root": str(args.data_root),
            "total_shards": len(tar_files),
            "total_samples": len(all_results),
            "valid_samples": len(valid),
            "error_samples": total_errors,
            "processing_time_seconds": round(elapsed, 1),
            "samples_per_second": round(len(all_results) / elapsed, 0) if elapsed > 0 else 0,
            "duration": {
                "min_seconds": round(min(durations), 2),
                "max_seconds": round(max(durations), 2),
                "mean_seconds": round(sum(durations) / len(durations), 2),
                "total_hours": round(sum(durations) / 3600, 1),
                "buckets": dict(dur_buckets),
            },
            "sample_rate": {
                "distribution": {str(k): v for k, v in sorted(sr_counts.items(), key=lambda x: -x[1])},
                "above_24khz": sum(1 for sr in sample_rates if sr >= 24000),
                "below_24khz": sum(1 for sr in sample_rates if sr < 24000),
            },
        }

        # Write summary JSON
        summary_path = args.out_dir / "peoples_speech_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total shards:    {len(tar_files):,}")
        print(f"Total samples:   {len(all_results):,}")
        print(f"Valid samples:   {len(valid):,}")
        print(f"Error samples:   {total_errors:,}")
        print()
        print("Duration:")
        print(f"  Min:   {min(durations):.2f}s")
        print(f"  Max:   {max(durations):.2f}s")
        print(f"  Mean:  {sum(durations)/len(durations):.2f}s")
        print(f"  Total: {sum(durations)/3600:.1f} hours")
        print()
        print("Duration buckets:")
        for bucket in ["<1s", "1-5s", "5-10s", "10-30s", "30-60s", "1-2min", "2-5min", ">5min"]:
            if bucket in dur_buckets:
                pct = 100 * dur_buckets[bucket] / len(valid)
                print(f"  {bucket:>8}: {dur_buckets[bucket]:>8,} ({pct:>5.1f}%)")
        print()
        print("Sample rate distribution:")
        for sr, count in sorted(sr_counts.items(), key=lambda x: -x[1])[:10]:
            pct = 100 * count / len(valid)
            print(f"  {sr:>6} Hz: {count:>8,} ({pct:>5.1f}%)")
        print()
        print(f"Samples >= 24kHz: {summary['sample_rate']['above_24khz']:,}")
        print(f"Samples <  24kHz: {summary['sample_rate']['below_24khz']:,}")
        print()
        print(f"Processing time: {elapsed:.1f}s ({len(all_results)/elapsed:.0f} samples/s)")
        print(f"Output: {args.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
