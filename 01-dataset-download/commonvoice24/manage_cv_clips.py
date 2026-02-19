#!/usr/bin/env python3
"""Manage CommonVoice clips after Shar conversion.

Compresses clips for selected splits into per-split tar.zst archives,
and removes clips for other splits that have already been converted to Shar.

Archives preserve the ``clips/`` prefix so they can be extracted in-place::

    python -c "
    import tarfile, zstandard
    dctx = zstandard.ZstdDecompressor()
    with open('test_clips.tar.zst', 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode='r|') as tar:
                tar.extractall(path='/path/to/lang_dir')
    "

Usage:
    # Compress test/dev clips, delete train/other clips for Italian
    python manage_cv_clips.py \
        --lang_dir /iopsstor/scratch/cscs/xyixuan/audio-datasets/raw/commonvoice24/it \
        --compress test dev \
        --delete train other

    # Dry run
    python manage_cv_clips.py \
        --lang_dir /iopsstor/scratch/cscs/xyixuan/audio-datasets/raw/commonvoice24/it \
        --compress test dev \
        --delete train other \
        --dry_run
"""

import argparse
import csv
import logging
import tarfile
from pathlib import Path

import zstandard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def read_split_clips(lang_dir: Path, split: str) -> set[str]:
    """Read clip filenames referenced by a split's TSV."""
    tsv_path = lang_dir / f"{split}.tsv"
    if not tsv_path.is_file():
        logger.warning(f"TSV not found: {tsv_path}")
        return set()

    clips = set()
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            clips.add(row["path"])
    return clips


def compress_split(lang_dir: Path, split: str, clip_filenames: set[str],
                   dry_run: bool = False) -> None:
    """Compress clips for a split into a tar.zst archive."""
    clips_dir = lang_dir / "clips"
    archive_path = lang_dir / f"{split}_clips.tar.zst"

    if archive_path.is_file():
        logger.info(f"Archive already exists: {archive_path}, skipping")
        return

    existing = sorted(name for name in clip_filenames if (clips_dir / name).is_file())
    if not existing:
        logger.warning(f"No clips found on disk for split '{split}'")
        return

    logger.info(f"Compressing {len(existing):,} clips for '{split}' → {archive_path}")
    if dry_run:
        return

    cctx = zstandard.ZstdCompressor(level=3, threads=-1)
    with open(archive_path, "wb") as fh:
        with cctx.stream_writer(fh) as writer:
            with tarfile.open(fileobj=writer, mode="w|") as tar:
                for i, name in enumerate(existing):
                    tar.add(clips_dir / name, arcname=f"clips/{name}")
                    if (i + 1) % 10000 == 0:
                        logger.info(f"  [{split}] {i + 1:,}/{len(existing):,} clips")

    size_gb = archive_path.stat().st_size / 1e9
    logger.info(f"Created {archive_path} ({size_gb:.2f} GB)")


def delete_split_clips(lang_dir: Path, split: str, clip_filenames: set[str],
                       dry_run: bool = False) -> None:
    """Delete clips belonging to a split."""
    clips_dir = lang_dir / "clips"
    existing = [name for name in clip_filenames if (clips_dir / name).is_file()]

    if not existing:
        logger.info(f"No clips to delete for split '{split}'")
        return

    logger.info(f"Deleting {len(existing):,} clips for '{split}'")
    if dry_run:
        return

    for name in existing:
        (clips_dir / name).unlink()
    logger.info(f"Deleted {len(existing):,} clips for '{split}'")


def main():
    parser = argparse.ArgumentParser(
        description="Compress/delete CommonVoice clips per split",
    )
    parser.add_argument("--lang_dir", type=Path, required=True,
                        help="Path to a CV language dir (e.g. .../commonvoice24/it)")
    parser.add_argument("--compress", nargs="+", default=[],
                        help="Splits to compress into tar.zst (e.g. test dev)")
    parser.add_argument("--delete", nargs="+", default=[],
                        help="Splits whose clips to delete (e.g. train other)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would happen without making changes")

    args = parser.parse_args()

    if not args.lang_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {args.lang_dir}")

    clips_dir = args.lang_dir / "clips"
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"No clips directory: {clips_dir}")

    if args.dry_run:
        logger.info("=== DRY RUN ===")

    # Delete first (free space), then compress
    for split in args.delete:
        clips = read_split_clips(args.lang_dir, split)
        logger.info(f"Split '{split}': {len(clips):,} clips in TSV")
        delete_split_clips(args.lang_dir, split, clips, dry_run=args.dry_run)

    for split in args.compress:
        clips = read_split_clips(args.lang_dir, split)
        logger.info(f"Split '{split}': {len(clips):,} clips in TSV")
        compress_split(args.lang_dir, split, clips, dry_run=args.dry_run)

    # Clean up clips directory
    remaining = sum(1 for _ in clips_dir.iterdir())
    if remaining == 0 and not args.dry_run:
        clips_dir.rmdir()
        logger.info(f"Removed empty clips directory: {clips_dir}")
    else:
        logger.info(f"Remaining clips in {clips_dir}: {remaining:,}")


if __name__ == "__main__":
    main()
