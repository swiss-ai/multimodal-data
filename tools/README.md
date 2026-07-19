# Utility Tools

Standalone utilities for inspecting, verifying, and packaging tar, webdataset,
and parquet datasets. Each script is self-contained with its own `--help`.

## Counting and stats
- `count_samples.py` Count files inside each tar/tar.gz or parquet shard.
- `wds_stats.py` Compute statistics over webdataset shards.

## Integrity and verification
- `check_tar_integrity.py` Detect corrupted or empty tar shards.
- `verify_dataset.py` Full validation: image decode, UTF-8, key uniqueness.
- `verify_wds_samples.py` Spot-check decoded webdataset samples.
- `check_paths.py` Check that referenced paths exist.

## Duplicate inspection
- `check_duplicates.py`, `main_hashes.py` Extract images for specific hashes from tars.
- `main_top.py` Extract duplicate files grouped by hash.

## Sampling and inspection
- `extract_samples.py` Extract a few WebDataset keys per tar for review.
- `export_sample.py` Save full sample/segment JSON for inspection.
- `evaluate_quality.py` Score caption/sample quality.

## Text cleaning and quality
- `clean_text_prefix.py` Remove N leading lines from `.txt` members in a tar.
- `detect_text_repetition.py` Flag repetitive text in captions.
- `search_prompt_leak.py` Search webdataset tars for prompt-leak phrases.
- `rouge_l.py` Token-level ROUGE-L via numba-JIT LCS.

## Bounding boxes
- `normalize_bbox.py` Normalize bounding-box coordinates.
- `verify_bbox.py` Verify bounding-box correctness.

## Packaging and misc
- `compress_dataset.py` Compress dataset directories to tar archives.
- `make_chunks.py` Split a JSONL file into per-worker chunks.
- `list_clean_targets.py` List shards eligible for a cleaning pass.
- `get_audio_file_from_ytc.py` Fetch audio files for selected languages.
