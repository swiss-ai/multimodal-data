#!/usr/bin/env python3
"""Repack MultiMed train parquets into a single dir with an explicit `locale`
column added per row.

Source schema (per language, in /raw/multimed/<Lang>/train-*.parquet):
    audio: struct<bytes, path>
    text: string
    duration: double

Output schema (per parquet in /processed/multimed/data/<lang_code>.parquet):
    audio: struct<bytes, path>
    text: string
    duration: double
    locale: string   ← added (zh|en|fr|de|vi)

Why a repack: language is encoded in the parent dir name, not as a column.
The convert pipeline needs `language_column` to carry per-row language
through to cut.supervisions[0].language. A small repack is cheaper than
sprouting per-language YAMLs / slurms.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC_ROOT = Path("/capstor/store/cscs/swissai/infra01/audio-datasets/raw/multimed")
DST_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/multimed/data"
)
LANG_DIR_TO_CODE = {
    "Chinese": "zh",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Vietnamese": "vi",
}


def pack_lang(lang_dir: str, lang_code: str) -> tuple[int, float]:
    files = sorted((SRC_ROOT / lang_dir).glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train shards under {SRC_ROOT / lang_dir}")
    tables = []
    for f in files:
        t = pq.read_table(f)
        n = t.num_rows
        t = t.append_column("locale", pa.array([lang_code] * n, type=pa.string()))
        tables.append(t)
    combined = pa.concat_tables(tables)
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    out = DST_ROOT / f"{lang_code}.parquet"
    pq.write_table(combined, out, compression="zstd", row_group_size=2000)
    total_dur = sum(combined.column("duration").to_pylist())
    return combined.num_rows, total_dur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", default=list(LANG_DIR_TO_CODE.keys()))
    args = parser.parse_args()

    grand_clips = 0
    grand_h = 0.0
    for lang_dir in args.langs:
        code = LANG_DIR_TO_CODE[lang_dir]
        clips, dur = pack_lang(lang_dir, code)
        grand_clips += clips
        grand_h += dur / 3600
        print(
            f"  {lang_dir} ({code}): {clips} clips, {dur/3600:.2f} h"
            f" -> {DST_ROOT / f'{code}.parquet'}"
        )
    print(f"Total: {grand_clips} clips, {grand_h:.2f} h")


if __name__ == "__main__":
    main()
