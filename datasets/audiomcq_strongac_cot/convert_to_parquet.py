#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


DEFAULT_RAW_DIR = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/sft/"
    "hf___Harland___AudioMCQ-StrongAC-GeminiCoT"
)
DEFAULT_OUT_DIR = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/sft/"
    "hf___Harland___AudioMCQ-StrongAC-GeminiCoT"
)

SCHEMA = pa.schema(
    [
        ("clip_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_id", pa.string()),
        ("question_type", pa.string()),
        ("audio_path", pa.string()),
        ("audio_bytes", pa.binary()),
        ("audio_format", pa.string()),
        ("duration_sec", pa.float64()),
        ("sampling_rate", pa.int64()),
        ("num_frames", pa.int64()),
        ("question", pa.string()),
        ("choices", pa.list_(pa.string())),
        ("answer", pa.string()),
        ("answer_index", pa.int64()),
        ("answer_label", pa.string()),
        ("gemini_cot", pa.string()),
        ("sft_prompt", pa.string()),
        ("sft_target", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AudioMCQ-StrongAC-GeminiCoT raw audio files to partitioned Parquet."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-rows", type=int, default=int(os.environ.get("SHARD_ROWS", "512")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "16")))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def audio_info(path: Path) -> tuple[float | None, int | None, int | None]:
    try:
        info = sf.info(str(path))
    except Exception:
        return None, None, None
    duration = float(info.duration) if info.duration is not None else None
    sr = int(info.samplerate) if info.samplerate else None
    frames = int(info.frames) if info.frames is not None else None
    return duration, sr, frames


def answer_index_and_label(row: dict) -> tuple[int | None, str | None]:
    choices = row.get("choices") or []
    answer = row.get("answer")
    try:
        index = choices.index(answer)
    except ValueError:
        return None, None
    return index, chr(65 + index)


def make_prompt(row: dict) -> str:
    choices = row.get("choices") or []
    choice_text = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    if choice_text:
        return f"{row.get('question', '')}\n\nChoices:\n{choice_text}"
    return row.get("question", "")


def flush(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    if not rows:
        return
    columns = {name: [row.get(name) for row in rows] for name in SCHEMA.names}
    writer.write_table(pa.Table.from_pydict(columns, schema=SCHEMA))
    rows.clear()


def row_to_parquet_row(raw_dir: Path, row: dict) -> tuple[dict | None, str | None, bool, str | None]:
    rel_audio_path = row["audio_path"]
    audio_path = raw_dir / rel_audio_path
    if not audio_path.exists():
        return None, rel_audio_path, False, None

    duration, sr, frames = audio_info(audio_path)
    source_dataset = row.get("source_dataset")
    source_id = str(row.get("id"))
    question_type = str(row.get("question_type"))
    clip_id = f"{question_type}_{source_dataset}_{source_id}" if source_dataset else f"{question_type}_{source_id}"
    answer_index, answer_label = answer_index_and_label(row)

    return (
        {
            "clip_id": clip_id,
            "source_dataset": source_dataset,
            "source_id": source_id,
            "question_type": question_type,
            "audio_path": rel_audio_path,
            "audio_bytes": audio_path.read_bytes(),
            "audio_format": audio_path.suffix.lstrip(".").lower(),
            "duration_sec": duration,
            "sampling_rate": sr,
            "num_frames": frames,
            "question": row.get("question"),
            "choices": row.get("choices") or [],
            "answer": row.get("answer"),
            "answer_index": answer_index,
            "answer_label": answer_label,
            "gemini_cot": row.get("gemini_cot"),
            "sft_prompt": make_prompt(row),
            "sft_target": row.get("answer"),
        },
        None,
        duration is None,
        source_dataset,
    )


def write_shard(args: tuple[Path, Path, str, int, int, list[dict]]) -> dict:
    raw_dir, data_dir, partition, shard_idx, num_shards, shard_input = args
    shard_name = f"{partition}-train-{shard_idx:05d}-of-{num_shards:05d}.parquet"
    shard_path = data_dir / shard_name
    tmp_path = data_dir / f"{shard_name}.tmp"
    rows: list[dict] = []
    missing_audio: list[str] = []
    duration_missing = 0
    source_counts = Counter()
    written = 0

    if tmp_path.exists():
        tmp_path.unlink()

    with pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd") as writer:
        for input_row in shard_input:
            output_row, missing, missing_duration, source_dataset = row_to_parquet_row(raw_dir, input_row)
            if missing:
                missing_audio.append(missing)
                continue
            if missing_duration:
                duration_missing += 1
            if source_dataset:
                source_counts[source_dataset] += 1
            rows.append(output_row)
            written += 1

            if len(rows) >= 32:
                flush(writer, rows)
        flush(writer, rows)

    os.replace(tmp_path, shard_path)
    return {
        "partition": partition,
        "shard_path": str(shard_path),
        "input_rows": len(shard_input),
        "written": written,
        "missing_audio": missing_audio,
        "duration_missing": duration_missing,
        "source_counts": dict(source_counts),
    }


def load_rows(raw_dir: Path) -> dict[str, list[dict]]:
    data_jsonl = raw_dir / "data.jsonl"
    if not data_jsonl.exists():
        raise FileNotFoundError(data_jsonl)

    partitions: dict[str, list[dict]] = defaultdict(list)
    with data_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            question_type = row.get("question_type")
            if not question_type:
                raise ValueError(f"Missing question_type in row: {row}")
            partitions[str(question_type)].append(row)
    return dict(sorted(partitions.items()))


def prepare_output(out_dir: Path, overwrite: bool) -> Path:
    data_dir = out_dir / "data"
    if overwrite and out_dir.exists():
        raise RuntimeError(
            "Refusing to recursively delete output from this script. "
            f"Clean it manually first: {out_dir}"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(data_dir.glob("*.parquet"))
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing parquet shards in {data_dir}: {existing[:3]}"
        )
    for tmp in data_dir.glob("*.tmp"):
        tmp.unlink()
    return data_dir


def main() -> None:
    args = parse_args()
    if args.shard_rows <= 0:
        raise ValueError("--shard-rows must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")

    partitions = load_rows(args.raw_dir)
    data_dir = prepare_output(args.out_dir, args.overwrite)

    shard_specs = []
    expected_counts = {partition: len(rows) for partition, rows in partitions.items()}
    partition_shards = {}
    for partition, rows in partitions.items():
        num_shards = (len(rows) + args.shard_rows - 1) // args.shard_rows
        partition_shards[partition] = num_shards
        for shard_idx in range(num_shards):
            shard_specs.append(
                (
                    args.raw_dir,
                    data_dir,
                    partition,
                    shard_idx,
                    num_shards,
                    rows[shard_idx * args.shard_rows : (shard_idx + 1) * args.shard_rows],
                )
            )

    max_workers = max(1, min(args.num_workers, len(shard_specs)))
    print(
        f"Writing {sum(expected_counts.values())} rows across "
        f"{len(shard_specs)} shards with {max_workers} workers",
        flush=True,
    )
    print(f"Partition counts: {expected_counts}", flush=True)

    counts = Counter()
    source_counts: dict[str, Counter] = defaultdict(Counter)
    missing_audio: list[str] = []
    duration_missing = Counter()
    shard_paths: list[str] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(write_shard, spec): spec for spec in shard_specs}
        for future in as_completed(futures):
            result = future.result()
            partition = result["partition"]
            counts[partition] += result["written"]
            duration_missing[partition] += result["duration_missing"]
            missing_audio.extend(result["missing_audio"])
            shard_paths.append(result["shard_path"])
            for source_dataset, count in result["source_counts"].items():
                source_counts[partition][source_dataset] += count
            print(
                f"Wrote {Path(result['shard_path']).name}: "
                f"{result['written']}/{result['input_rows']} rows",
                flush=True,
            )

    shard_paths.sort()
    manifest = {
        "name": "hf___Harland___AudioMCQ-StrongAC-GeminiCoT",
        "source_raw_dir": str(args.raw_dir),
        "source_repo_id": "Harland/AudioMCQ-StrongAC-GeminiCoT",
        "source_revision": "cb33687ce4dc3dace4e203a2dd584fa46ae312da",
        "data_dir": str(data_dir),
        "parquet_globs": {
            partition: f"{partition}-train-*.parquet"
            for partition in expected_counts
        },
        "partition_counts": dict(sorted(counts.items())),
        "expected_partition_counts": expected_counts,
        "partition_shards": partition_shards,
        "shard_rows": args.shard_rows,
        "num_workers": max_workers,
        "parquet_files": shard_paths,
        "schema": SCHEMA.names,
        "source_dataset_counts": {
            partition: dict(sorted(counter.items()))
            for partition, counter in sorted(source_counts.items())
        },
        "duration_missing_count": dict(sorted(duration_missing.items())),
        "missing_audio_count": len(missing_audio),
        "missing_audio": missing_audio[:50],
        "license_note": (
            "Converted from AudioMCQ-StrongAC-GeminiCoT. Dataset card is Apache-2.0; "
            "embedded audio has mixed upstream provenance."
        ),
    }
    if dict(sorted(counts.items())) != expected_counts:
        raise RuntimeError(f"Partition count mismatch: expected={expected_counts}, actual={dict(counts)}")
    if missing_audio:
        raise RuntimeError(f"Missing {len(missing_audio)} audio files; examples={missing_audio[:5]}")

    (args.out_dir / "_SOURCE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "README.md").write_text(
        "# AudioMCQ-StrongAC-GeminiCoT Processed SFT Parquet\n\n"
        "Processed Parquet version of `Harland/AudioMCQ-StrongAC-GeminiCoT` "
        "with embedded `audio_bytes` and shards partitioned by `question_type`.\n\n"
        "Files live under `data/` and are named "
        "`{partition}-train-00000-of-000NN.parquet`, where partition is one of "
        "`speech`, `sound`, `music`, or `temporal`.\n\n"
        "Important columns: `clip_id`, `source_dataset`, `source_id`, "
        "`question_type`, `audio_path`, `audio_bytes`, `audio_format`, "
        "`duration_sec`, `sampling_rate`, `num_frames`, `question`, `choices`, "
        "`answer`, `answer_index`, `answer_label`, `gemini_cot`, `sft_prompt`, "
        "`sft_target`.\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
