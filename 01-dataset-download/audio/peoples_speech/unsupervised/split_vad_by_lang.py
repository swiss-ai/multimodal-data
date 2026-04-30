#!/usr/bin/env python3
"""Split VAD per-shard JSONL files by language.

Reads all VAD JSONL files and writes per-language versions.

Input:  vad_results_all_per_shard/audio2_005047.jsonl  (mixed languages)
Output: vad_per_lang/en/audio2_005047.jsonl            (English only)
        vad_per_lang/es/audio2_005047.jsonl            (Spanish only)
        ...

Usage:
    python split_vad_by_lang.py \
        --vad-dir /path/to/vad_results_all_per_shard \
        --output-dir /path/to/vad_per_lang \
        --num-workers 64
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def process_one_file(args):
    jsonl_path, output_dir = args
    fname = Path(jsonl_path).name

    lang_entries = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            val = list(d.values())[0]
            lang = val.get("lang", "unknown")
            lang_entries[lang].append(line)

    for lang, lines in lang_entries.items():
        lang_dir = Path(output_dir) / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        with open(lang_dir / fname, "w") as f:
            f.write("\n".join(lines) + "\n")

    return {lang: len(lines) for lang, lines in lang_entries.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vad-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=64)
    args = parser.parse_args()

    jsonl_files = sorted(Path(args.vad_dir).glob("*.jsonl"))
    logger.info(f"Found {len(jsonl_files)} JSONL files")

    work_items = [(str(f), args.output_dir) for f in jsonl_files]

    from collections import Counter
    total_langs = Counter()

    with Pool(processes=args.num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one_file, work_items, chunksize=1)):
            for lang, count in result.items():
                total_langs[lang] += count
            if (i + 1) % 500 == 0:
                logger.info(f"{i+1}/{len(jsonl_files)} files processed")

    logger.info(f"Done. Languages:")
    for lang, count in total_langs.most_common():
        logger.info(f"  {lang}: {count:,} entries")


if __name__ == "__main__":
    main()
