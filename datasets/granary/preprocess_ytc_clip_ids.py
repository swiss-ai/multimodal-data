#!/usr/bin/env python3
"""Preprocess YTC JSONL to add dense clip_num and source_id.

Reads uttid_text.jsonl.gz per language, parses video ID and timestamp
from the uttid, assigns dense clip_num per video (sorted by timestamp),
and breaks runs when gap exceeds --max-gap-sec.

Output: uttid_text_with_clipnum.jsonl.gz with added fields:
  - source_id: video ID (with _R{n} suffix if gap-broken)
  - clip_num: dense 0-based index within each source_id
  - clip_start_sec: start time in seconds

Usage:
    python preprocess_ytc_clip_ids.py \
        --input-dir /capstor/.../ytc-granary/asr \
        --max-gap-sec 30
"""

import argparse
import gzip
import os
import time
from multiprocessing import Pool
from pathlib import Path

import orjson


def _process_lang(args):
    lang_dir_str, max_gap_cs = args
    lang_dir = Path(lang_dir_str)
    input_path = lang_dir / "uttid_text.jsonl.gz"
    if not input_path.is_file():
        return None

    rows = []
    with gzip.open(input_path, "rb") as f:
        for line in f:
            d = orjson.loads(line)
            parts = d["uttid"].rsplit("-", 2)
            vid = parts[0]
            start_cs = int(parts[1])
            dur_cs = int(parts[2])
            rows.append((vid, start_cs, dur_cs, d))

    rows.sort(key=lambda x: (x[0], x[1]))

    prev_vid = None
    prev_end_cs = 0
    run_idx = 0
    clip_num = 0

    output_path = lang_dir / "uttid_text_with_clipnum.jsonl.gz"
    with gzip.open(output_path, "wb") as f:
        for vid, start_cs, dur_cs, d in rows:
            if vid != prev_vid:
                run_idx = 0
                clip_num = 0
            elif start_cs - prev_end_cs > max_gap_cs:
                run_idx += 1
                clip_num = 0

            source_id = f"{vid}_R{run_idx}" if run_idx > 0 else vid
            d["source_id"] = source_id
            d["clip_num"] = clip_num
            d["clip_start_sec"] = start_cs / 100.0
            f.write(orjson.dumps(d) + b"\n")

            prev_vid = vid
            prev_end_cs = start_cs + dur_cs
            clip_num += 1

    n_videos = len(set(r[0] for r in rows))
    return f"{lang_dir.name}: {len(rows):>8} entries, {n_videos:>6} videos"


def main():
    parser = argparse.ArgumentParser(description="Preprocess YTC clip IDs")
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Root dir containing per-language subdirs with uttid_text.jsonl.gz")
    parser.add_argument("--max-gap-sec", type=float, default=30.0,
                        help="Break runs when gap exceeds this (default: 30s)")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers (default: CPU count)")
    args = parser.parse_args()

    max_gap_cs = int(args.max_gap_sec * 100)
    lang_dirs = [(str(d), max_gap_cs) for d in sorted(args.input_dir.iterdir()) if d.is_dir()]

    n_workers = min(args.num_workers, len(lang_dirs))
    t0 = time.time()
    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_process_lang, lang_dirs):
            if result:
                print(result)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
