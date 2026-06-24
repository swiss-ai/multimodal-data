#!/usr/bin/env python3
"""Pack OpenSLR-108 MediaSpeech (.flac + .txt sidecar pairs) into per-language parquets.

Output schema (one row per clip):
    audio: struct<bytes: binary, path: string>
    transcription: string
    locale: string  (lang code: ar/es/fr/tr)
    duration_ss: float

Source: /capstor/.../raw/openslr108___mediaspeech/{AR,ES,FR,TR}/*.{flac,txt}
Dest:   /capstor/.../processed/openslr108_mediaspeech/data/{lang}.parquet
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

SRC_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/openslr108___mediaspeech"
)
DST_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/openslr108_mediaspeech/data"
)
LANGS = ("AR", "ES", "FR", "TR")


def pack_lang(lang: str) -> tuple[int, float]:
    src_dir = SRC_ROOT / lang
    flacs = sorted(src_dir.glob("*.flac"))
    rows_audio: list[dict] = []
    rows_text: list[str] = []
    rows_dur: list[float] = []
    rows_locale: list[str] = []
    total_dur = 0.0
    skipped = 0
    for flac in flacs:
        txt = flac.with_suffix(".txt")
        if not txt.is_file():
            skipped += 1
            continue
        audio_bytes = flac.read_bytes()
        try:
            info = sf.info(io.BytesIO(audio_bytes))
        except Exception:
            skipped += 1
            continue
        rows_audio.append({"bytes": audio_bytes, "path": flac.name})
        rows_text.append(txt.read_text(encoding="utf-8").strip())
        rows_dur.append(info.duration)
        rows_locale.append(lang.lower())
        total_dur += info.duration

    table = pa.table(
        {
            "audio": pa.array(
                rows_audio,
                type=pa.struct(
                    [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
                ),
            ),
            "transcription": pa.array(rows_text, type=pa.string()),
            "duration_ss": pa.array(rows_dur, type=pa.float32()),
            "locale": pa.array(rows_locale, type=pa.string()),
        }
    )

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    out = DST_ROOT / f"{lang.lower()}.parquet"
    pq.write_table(table, out, compression="zstd", row_group_size=2000)
    return len(rows_audio), total_dur


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", default=list(LANGS))
    args = parser.parse_args()

    grand_clips = 0
    grand_h = 0.0
    for lang in args.langs:
        clips, dur = pack_lang(lang)
        grand_clips += clips
        grand_h += dur / 3600
        print(f"  {lang}: {clips} clips, {dur/3600:.2f} h -> {DST_ROOT / f'{lang.lower()}.parquet'}")
    print(f"Total: {grand_clips} clips, {grand_h:.2f} h")


if __name__ == "__main__":
    main()
