#!/usr/bin/env python3
"""Build VoiceAssistant-400K SFT conversation parquets.

gpt-omni's VoiceAssistant-400K is consumed for voice-in/text-out SFT:

  * Drop split ``identity`` (4,306 model-identity rows that would poison
    Apertus self-identity training).
  * Drop ``answer_snac`` (SNAC codec, incompatible with our WavTokenizer cache).
  * Group rows by ``index``, sort by ``round``, emit one conversation per index.
    ``rlhf`` is multi-turn (rounds 0..33); other 5 splits are single-turn.

Two outputs under ``processed/sft/VoiceAssistant-400K/`` (no ``/examples/``
prefix, matches the post-HeySQuAD convention):

  data/train-NNNNN-of-NNNNN.parquet  -- flat per-turn audio + metadata for the
                                        SHAR convert stage. audio_id is
                                        f"voiceassistant_{{index}}_{{round}}".
  conversations/train.parquet        -- one row per conversation with
                                        messages_json: <audio> user turns
                                        alternating with text assistant turns.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


DEFAULT_RAW_DIR = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/sft/VoiceAssistant-400K"
)
DEFAULT_OUT_DIR = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/sft/VoiceAssistant-400K"
)
EXCLUDED_SPLITS = ("identity",)
AUDIO_PLACEHOLDER = "<audio>"


def parse_wav_metadata(audio_bytes: bytes) -> dict:
    """Header-only decode via soundfile (handles IEEE-float WAV; stdlib `wave` rejects format=3)."""
    info = sf.info(io.BytesIO(audio_bytes))
    return {
        "sampling_rate": int(info.samplerate),
        "num_frames": int(info.frames),
        "duration_sec": float(info.duration),
    }


def _round_str(round_field) -> str:
    """Normalize round to canonical string. Single-turn splits coerce to '0'."""
    if round_field in (None, "None"):
        return "0"
    return str(round_field)


def _conv_id(split_name: str, index: str) -> str:
    """Per-conversation stable id. Note: `index` is NOT unique across splits
    (e.g. trivia_*_choice and qa_assistant_v* both restart at 000000000), so
    split_name must be part of the key to avoid collisions."""
    return f"voiceassistant_{split_name}_{index}"


def _audio_id(split_name: str, index: str, round_norm: str) -> str:
    return f"{_conv_id(split_name, index)}_{round_norm}"


def process(*, raw_dir: Path, out_dir: Path) -> tuple[int, int, int]:
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "conversations").mkdir(parents=True, exist_ok=True)

    in_files = sorted(glob.glob(str(raw_dir / "data") + "/train-*.parquet"))
    if not in_files:
        raise FileNotFoundError(f"no parquets under {raw_dir / 'data'}")
    print(f"[voiceassistant] {len(in_files)} input shards", flush=True)

    conversations: dict[tuple[str, str], list[dict]] = defaultdict(list)
    total_turns = 0
    skipped_identity = 0
    t0 = time.time()

    for i, path in enumerate(in_files):
        df = pl.read_parquet(
            path,
            columns=[
                "split_name", "index", "round",
                "question", "answer", "question_audio",
            ],
        )
        before = len(df)
        df = df.filter(~pl.col("split_name").is_in(list(EXCLUDED_SPLITS)))
        skipped_identity += before - len(df)

        rows_for_data = []
        for row in df.iter_rows(named=True):
            r_norm = _round_str(row["round"])
            aid = _audio_id(row["split_name"], row["index"], r_norm)
            ab = row["question_audio"]["bytes"]
            meta = parse_wav_metadata(ab)
            rows_for_data.append({
                "audio_id": aid,
                "audio_bytes": ab,
                "audio_format": "wav",
                "audio_path": row["question_audio"]["path"],
                "duration_sec": meta["duration_sec"],
                "sampling_rate": meta["sampling_rate"],
                "num_frames": meta["num_frames"],
                "split_name": row["split_name"],
                "index": row["index"],
                "round": r_norm,
                "question_text": row["question"],
                "answer_text": row["answer"],
            })
            conversations[(row["index"], row["split_name"])].append({
                "round": int(r_norm),
                "answer": row["answer"],
                "audio_id": aid,
            })

        if rows_for_data:
            out_data = (
                out_dir / "data"
                / f"train-{i:05d}-of-{len(in_files):05d}.parquet"
            )
            # Write via pyarrow, not Polars: Polars's write_parquet produced
            # silently-unreadable parquets in two prior runs (shards 199-232
            # contiguous, ~10% of total) when rows contained large audio
            # values (>8 MB per row, max ~23 MB for 4-min TTS clips).
            # pyarrow-written parquets are guaranteed pyarrow-readable.
            table = pa.Table.from_pylist(rows_for_data)
            pq.write_table(
                table,
                str(out_data),
                compression="snappy",
                row_group_size=500,
            )

        total_turns += len(rows_for_data)
        if (i + 1) % 25 == 0 or i + 1 == len(in_files):
            print(
                f"  [voiceassistant] shard {i+1}/{len(in_files)}  "
                f"turns={total_turns:,} convs={len(conversations):,}  "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    print(
        f"[voiceassistant] building messages_json for "
        f"{len(conversations):,} conversations",
        flush=True,
    )
    conv_rows = []
    per_split_conversations: dict[str, int] = defaultdict(int)
    per_split_turns: dict[str, int] = defaultdict(int)
    for (index, split_name), turns in conversations.items():
        turns.sort(key=lambda t: t["round"])
        messages: list[dict] = []
        audio_ids: list[str] = []
        for turn in turns:
            messages.append({"role": "user", "content": AUDIO_PLACEHOLDER})
            messages.append({"role": "assistant", "content": turn["answer"]})
            audio_ids.append(turn["audio_id"])
        conv_rows.append({
            "sample_id": _conv_id(split_name, index),
            "audio_ids": audio_ids,
            "messages_json": json.dumps(messages, ensure_ascii=False),
            "source_dataset": "voiceassistant_400k",
            "split_name": split_name,
            "num_turns": len(turns),
        })
        per_split_conversations[split_name] += 1
        per_split_turns[split_name] += len(turns)

    conv_df = pl.DataFrame(conv_rows)
    conv_df.write_parquet(str(out_dir / "conversations" / "train.parquet"))

    manifest = {
        "format": "audio_voice_in_text_out",
        "schema_version": 1,
        "n_conversations": len(conv_df),
        "n_turns": total_turns,
        "n_skipped_identity": skipped_identity,
        "user_template": AUDIO_PLACEHOLDER,
        "assistant_template": "{answer}",
        "excluded_splits": list(EXCLUDED_SPLITS),
        "per_split_conversations": dict(sorted(per_split_conversations.items())),
        "per_split_turns": dict(sorted(per_split_turns.items())),
    }
    with open(out_dir / "conversations" / "_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(
        f"[voiceassistant] DONE: {total_turns:,} turns -> {len(conv_df):,} "
        f"conversations ({skipped_identity:,} identity rows excluded), "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return total_turns, len(conv_df), skipped_identity


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)
    process(raw_dir=args.raw_dir, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
