"""Pack AIDC-AI/Marco_Longspeech into normalized audio-SFT Parquet.

The raw dataset has many QA examples pointing at a smaller set of long WAV
files. Embedding audio bytes in every example row would duplicate large audio
payloads, so this script writes two Parquet families:

    output/
      _SOURCE.json
      _MANIFEST.json
      examples/
        train-00000.parquet
        val-00000.parquet
        test-00000.parquet
      media/
        audio/
          audio-00000-of-NNNNN.parquet
          ...
          _index.parquet

``media/audio`` contains one row per unique audio file, with embedded
compressed audio bytes. Examples contain one row per SFT sample and reference
audio by stable ``audio_id``.

Default audio policy is FLAC for Marco because the source files are PCM WAV.
Use --audio-codec copy to keep original WAV bytes if exact container
preservation matters more than storage size.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
import wave
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_RAW_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/sft/"
    "hf___AIDC-AI___Marco_Longspeech"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/sft/"
    "hf___AIDC-AI___Marco_Longspeech"
)

DATASET_ID = "hf___AIDC-AI___Marco_Longspeech"
SOURCE_REPO = "https://huggingface.co/datasets/AIDC-AI/Marco_Longspeech"


AUDIO_REF_T = pa.list_(
    pa.struct([
        ("audio_id", pa.string()),
        ("source_path", pa.string()),
    ])
)

MESSAGE_T = pa.list_(
    pa.struct([
        ("role", pa.string()),
        ("content", pa.string()),
        ("audio", AUDIO_REF_T),
    ])
)

EXAMPLE_SCHEMA = pa.schema([
    ("sample_id", pa.string()),
    ("dataset_id", pa.string()),
    ("split", pa.string()),
    ("task_dir", pa.string()),
    ("task", pa.string()),
    ("language", pa.string()),
    ("messages", MESSAGE_T),
    ("audio_ids", pa.list_(pa.string())),
    ("metadata_json", pa.string()),
    ("source_json", pa.string()),
    ("source_jsonl", pa.string()),
    ("source_line", pa.int32()),
])

ASSET_SCHEMA = pa.schema([
    ("audio_id", pa.string()),
    ("audio", pa.struct([
        ("bytes", pa.binary()),
        ("path", pa.string()),
    ])),
    ("header", pa.struct([
        ("format", pa.string()),
        ("codec", pa.string()),
        ("sample_rate", pa.int32()),
        ("channels", pa.int32()),
        ("duration_sec", pa.float64()),
        ("num_frames", pa.int64()),
        ("sample_width_bytes", pa.int32()),
    ])),
    ("source", pa.struct([
        ("relative_path", pa.string()),
        ("sha256", pa.string()),
        ("original_size_bytes", pa.int64()),
        ("original_format", pa.string()),
    ])),
])

MEDIA_AUDIO_INDEX_SCHEMA = pa.schema([
    ("audio_id", pa.string()),
    ("shard_path", pa.string()),
    ("row_index", pa.int64()),
    ("source_relative_path", pa.string()),
    ("sha256", pa.string()),
    ("duration_sec", pa.float64()),
    ("sample_rate", pa.int32()),
    ("channels", pa.int32()),
    ("format", pa.string()),
    ("codec", pa.string()),
    ("original_size_bytes", pa.int64()),
    ("encoded_size_bytes", pa.int64()),
])


def audio_id_from_path(relative_path: str) -> str:
    stem = Path(relative_path).with_suffix("").as_posix()
    safe = stem.replace("/", "__").replace(" ", "_")
    return f"marco_longspeech__{safe}"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_wav_header(path: Path) -> dict[str, int | float | str]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()
        sample_width_bytes = wav.getsampwidth()
    duration_sec = num_frames / sample_rate if sample_rate else 0.0
    return {
        "format": "wav",
        "codec": "pcm_s16le" if sample_width_bytes == 2 else "pcm",
        "sample_rate": sample_rate,
        "channels": channels,
        "duration_sec": duration_sec,
        "num_frames": num_frames,
        "sample_width_bytes": sample_width_bytes,
    }


def encode_audio(path: Path, codec: str) -> tuple[bytes, dict[str, int | float | str]]:
    source_header = read_wav_header(path)
    if codec == "copy":
        return path.read_bytes(), source_header

    if codec != "flac":
        raise ValueError(f"unsupported audio codec: {codec}")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required for --audio-codec flac") from exc

    out = io.BytesIO()
    with sf.SoundFile(str(path), mode="r") as src:
        with sf.SoundFile(
            out,
            mode="w",
            samplerate=src.samplerate,
            channels=src.channels,
            format="FLAC",
            subtype="PCM_16",
        ) as dst:
            while True:
                block = src.read(1_048_576, dtype="int16", always_2d=True)
                if len(block) == 0:
                    break
                dst.write(block)
    header = dict(source_header)
    header["format"] = "flac"
    header["codec"] = "flac"
    return out.getvalue(), header


def message_audio_values(message: dict[str, Any]) -> list[str]:
    raw_audio = message.get("audio")
    if raw_audio is None:
        return []
    if isinstance(raw_audio, str):
        return [raw_audio]
    if isinstance(raw_audio, list):
        values: list[str] = []
        for item in raw_audio:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                candidate = item.get("path") or item.get("audio") or item.get("source_path")
                if isinstance(candidate, str):
                    values.append(candidate)
        return values
    if isinstance(raw_audio, dict):
        candidate = raw_audio.get("path") or raw_audio.get("audio") or raw_audio.get("source_path")
        return [candidate] if isinstance(candidate, str) else []
    return []


def normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    audio_ids: list[str] = []

    for message in messages:
        refs = []
        for source_path in message_audio_values(message):
            audio_id = audio_id_from_path(source_path)
            refs.append({"audio_id": audio_id, "source_path": source_path})
            audio_ids.append(audio_id)
        normalized.append({
            "role": str(message.get("role") or ""),
            "content": str(message.get("content") or ""),
            "audio": refs,
        })

    return normalized, audio_ids


def iter_jsonl_files(raw_root: Path) -> list[Path]:
    qa_root = raw_root / "LongSpeechQA"
    files = sorted(qa_root.glob("*/*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no LongSpeechQA jsonl files found under {qa_root}")
    return files


def read_examples(
    raw_root: Path,
    max_examples_per_jsonl: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    examples: list[dict[str, Any]] = []
    audio_paths: set[str] = set()

    for jsonl_path in iter_jsonl_files(raw_root):
        task_dir = jsonl_path.parent.name
        split = jsonl_path.stem
        rel_jsonl = jsonl_path.relative_to(raw_root).as_posix()

        with jsonl_path.open() as f:
            for line_idx, line in enumerate(f):
                if max_examples_per_jsonl is not None and line_idx >= max_examples_per_jsonl:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                raw_messages = row.get("messages") or []
                messages, row_audio_ids = normalize_messages(raw_messages)
                for message in messages:
                    for ref in message["audio"]:
                        audio_paths.add(ref["source_path"])

                metadata = {
                    k: v for k, v in row.items()
                    if k not in {"language", "task", "messages"}
                }
                sample_id = f"marco_longspeech__{split}__{task_dir}__{line_idx:08d}"
                examples.append({
                    "sample_id": sample_id,
                    "dataset_id": DATASET_ID,
                    "split": split,
                    "task_dir": task_dir,
                    "task": str(row.get("task") or task_dir),
                    "language": str(row.get("language") or ""),
                    "messages": messages,
                    "audio_ids": row_audio_ids,
                    "metadata_json": stable_json(metadata),
                    "source_json": stable_json(row),
                    "source_jsonl": rel_jsonl,
                    "source_line": line_idx + 1,
                })

    return examples, sorted(audio_paths)


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def write_examples(
    examples: list[dict[str, Any]],
    output_dir: Path,
    rows_per_shard: int,
    compression: str,
    overwrite: bool,
) -> dict[str, Any]:
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_split.setdefault(example["split"], []).append(example)

    split_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}

    for split, rows in sorted(by_split.items()):
        split_counts[split] = len(rows)
        shards = chunked(rows, rows_per_shard)
        for shard_idx, shard in enumerate(shards):
            out_path = examples_dir / f"{split}-{shard_idx:05d}.parquet"
            for row in shard:
                task_counts[row["task_dir"]] = task_counts.get(row["task_dir"], 0) + 1
            if out_path.exists() and not overwrite:
                print(f"skip existing examples shard: {out_path}", flush=True)
                continue
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            table = pa.Table.from_pylist(shard, schema=EXAMPLE_SCHEMA)
            pq.write_table(table, tmp_path, compression=compression)
            tmp_path.replace(out_path)
            print(f"wrote {out_path} ({len(shard):,} rows)", flush=True)

    return {
        "num_examples": len(examples),
        "splits": split_counts,
        "tasks": task_counts,
        "num_example_shards": sum(math.ceil(len(rows) / rows_per_shard) for rows in by_split.values()),
    }


def pack_media_audio_shard(args: tuple[int, int, list[str], str, str, str, int, bool]) -> dict[str, Any]:
    (
        shard_idx,
        num_shards,
        rel_paths,
        raw_root_s,
        output_dir_s,
        audio_codec,
        write_batch_size,
        overwrite,
    ) = args
    raw_root = Path(raw_root_s)
    output_dir = Path(output_dir_s)
    media_audio_dir = output_dir / "media" / "audio"
    media_audio_dir.mkdir(parents=True, exist_ok=True)
    out_path = media_audio_dir / f"audio-{shard_idx:05d}-of-{num_shards:05d}.parquet"
    shard_path = out_path.relative_to(output_dir).as_posix()
    if out_path.exists() and not overwrite:
        return {
            "shard_idx": shard_idx,
            "written": 0,
            "failed": 0,
            "bytes_in": 0,
            "bytes_out": 0,
            "index_rows": [],
            "failures": [],
            "skipped_existing": True,
        }

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    written = 0
    failed = 0
    bytes_in = 0
    bytes_out = 0
    batch: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str | int]] = []

    with pq.ParquetWriter(tmp_path, ASSET_SCHEMA, compression="zstd") as writer:
        def flush() -> None:
            nonlocal batch
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=ASSET_SCHEMA))
                batch = []

        for rel_path in rel_paths:
            source_path = raw_root / rel_path
            if not source_path.is_file():
                failed += 1
                failures.append({
                    "shard_idx": shard_idx,
                    "source_relative_path": rel_path,
                    "error_type": "FileNotFoundError",
                    "error": "referenced audio file is missing",
                })
                continue

            try:
                original_size = source_path.stat().st_size
                source_hash = sha256_file(source_path)
                encoded_bytes, header = encode_audio(source_path, audio_codec)
            except Exception as exc:
                failed += 1
                failures.append({
                    "shard_idx": shard_idx,
                    "source_relative_path": rel_path,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                })
                continue

            audio_id = audio_id_from_path(rel_path)
            row = {
                "audio_id": audio_id,
                "audio": {
                    "bytes": encoded_bytes,
                    "path": rel_path,
                },
                "header": header,
                "source": {
                    "relative_path": rel_path,
                    "sha256": source_hash,
                    "original_size_bytes": original_size,
                    "original_format": source_path.suffix.lstrip(".").lower(),
                },
            }
            batch.append(row)
            written += 1
            bytes_in += original_size
            bytes_out += len(encoded_bytes)
            row_index = written - 1
            index_rows.append({
                "audio_id": audio_id,
                "shard_path": shard_path,
                "row_index": row_index,
                "source_relative_path": rel_path,
                "sha256": source_hash,
                "duration_sec": float(header["duration_sec"]),
                "sample_rate": int(header["sample_rate"]),
                "channels": int(header["channels"]),
                "format": str(header["format"]),
                "codec": str(header["codec"]),
                "original_size_bytes": original_size,
                "encoded_size_bytes": len(encoded_bytes),
            })

            if len(batch) >= write_batch_size:
                flush()
        flush()

    tmp_path.replace(out_path)
    return {
        "shard_idx": shard_idx,
        "written": written,
        "failed": failed,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "index_rows": index_rows,
        "failures": failures,
        "skipped_existing": False,
    }


def write_media_audio(
    audio_paths: list[str],
    raw_root: Path,
    output_dir: Path,
    rows_per_shard: int,
    num_workers: int,
    audio_codec: str,
    write_batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    media_audio_dir = output_dir / "media" / "audio"
    media_audio_dir.mkdir(parents=True, exist_ok=True)
    shards = chunked(audio_paths, rows_per_shard)
    num_shards = len(shards)
    tasks = [
        (
            idx,
            num_shards,
            shard,
            str(raw_root),
            str(output_dir),
            audio_codec,
            write_batch_size,
            overwrite,
        )
        for idx, shard in enumerate(shards)
    ]
    workers = max(1, min(num_workers, len(tasks)))
    print(
        f"packing {len(audio_paths):,} unique media/audio rows into {num_shards:,} shards "
        f"with {workers} workers, codec={audio_codec}",
        flush=True,
    )

    total_written = 0
    total_failed = 0
    total_bytes_in = 0
    total_bytes_out = 0
    index_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str | int]] = []
    skipped_existing = 0
    started = time.time()

    with Pool(workers) as pool:
        for result in pool.imap_unordered(pack_media_audio_shard, tasks):
            if result["skipped_existing"]:
                print(f"skip existing media/audio shard {result['shard_idx']:05d}", flush=True)
                skipped_existing += 1
                continue
            total_written += result["written"]
            total_failed += result["failed"]
            total_bytes_in += result["bytes_in"]
            total_bytes_out += result["bytes_out"]
            index_rows.extend(result["index_rows"])
            failures.extend(result["failures"])
            ratio = total_bytes_out / total_bytes_in if total_bytes_in else 0.0
            elapsed = time.time() - started
            print(
                f"media/audio shard {result['shard_idx']:05d}: "
                f"{result['written']:,} written, {result['failed']:,} failed; "
                f"total={total_written:,}, ratio={ratio:.3f}, elapsed={elapsed/60:.1f}m",
                flush=True,
            )

    if failures:
        failures_path = media_audio_dir / "_failures.jsonl"
        with failures_path.open("w") as f:
            for failure in failures:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise RuntimeError(
            f"{total_failed:,} media/audio rows failed to pack; see {failures_path}"
        )

    index_path = media_audio_dir / "_index.parquet"
    if skipped_existing and index_path.exists() and not overwrite:
        print(f"kept existing media/audio index: {index_path}", flush=True)
    elif skipped_existing:
        raise RuntimeError(
            "some media/audio shards already existed but _index.parquet was missing; "
            "rerun with --overwrite to rebuild a consistent index"
        )
    else:
        index_rows.sort(key=lambda row: row["audio_id"])
        tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
        table = pa.Table.from_pylist(index_rows, schema=MEDIA_AUDIO_INDEX_SCHEMA)
        pq.write_table(table, tmp_path, compression="zstd")
        tmp_path.replace(index_path)
        print(f"wrote {index_path} ({len(index_rows):,} rows)", flush=True)

    return {
        "num_audio": len(audio_paths),
        "num_audio_shards": num_shards,
        "num_audio_written": total_written,
        "original_size_bytes": total_bytes_in,
        "encoded_size_bytes": total_bytes_out,
        "compression_ratio": total_bytes_out / total_bytes_in if total_bytes_in else None,
    }


def write_source_manifest(raw_root: Path, output_dir: Path, args: argparse.Namespace) -> None:
    source = {
        "dataset_id": DATASET_ID,
        "repo_url": SOURCE_REPO,
        "raw_root": str(raw_root),
        "packed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "01-dataset-download/audio/marco_longspeech/pack_to_sft_parquet.py",
        "format": "normalized_audio_sft_parquet_v1",
        "audio_codec": args.audio_codec,
        "media_audio_rows_per_shard": args.media_audio_rows_per_shard,
        "example_rows_per_shard": args.example_rows_per_shard,
    }
    raw_source = raw_root / "_SOURCE.json"
    if raw_source.exists():
        source["raw_source"] = json.loads(raw_source.read_text())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")


def write_manifest(
    output_dir: Path,
    examples: list[dict[str, Any]],
    audio_paths: list[str],
    example_summary: dict[str, Any],
    media_summary: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    audio_set = set(audio_id_from_path(path) for path in audio_paths)
    referenced_audio: set[str] = set()
    task_stats: dict[str, dict[str, Any]] = {}

    for example in examples:
        task = example["task_dir"]
        stats = task_stats.setdefault(task, {"examples": 0, "unique_audio": set()})
        stats["examples"] += 1
        for audio_id in example["audio_ids"]:
            referenced_audio.add(audio_id)
            stats["unique_audio"].add(audio_id)

    missing_audio = sorted(referenced_audio - audio_set)
    unreferenced_audio = sorted(audio_set - referenced_audio)
    if missing_audio:
        raise RuntimeError(f"{len(missing_audio):,} referenced audio ids are missing from media/audio")

    manifest = {
        "dataset_id": DATASET_ID,
        "format": "normalized_audio_sft_parquet_v1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "layout": {
            "examples": "examples/*.parquet",
            "media_audio": "media/audio/audio-*.parquet",
            "media_audio_index": "media/audio/_index.parquet",
        },
        "num_examples": len(examples),
        "num_unique_audio": len(audio_paths),
        "splits": example_summary["splits"],
        "tasks": {
            task: {
                "examples": stats["examples"],
                "unique_audio": len(stats["unique_audio"]),
            }
            for task, stats in sorted(task_stats.items())
        },
        "shards": {
            "examples": example_summary["num_example_shards"],
            "media_audio": media_summary["num_audio_shards"],
        },
        "audio": {
            "codec": args.audio_codec,
            "original_size_bytes": media_summary["original_size_bytes"],
            "encoded_size_bytes": media_summary["encoded_size_bytes"],
            "compression_ratio": media_summary["compression_ratio"],
        },
        "invariants": {
            "examples_audio_ids_missing_from_media_audio": len(missing_audio),
            "media_audio_ids_unreferenced_by_examples": len(unreferenced_audio),
            "each_media_audio_id_stored_once": True,
        },
    }
    out_path = output_dir / "_MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--audio-codec", choices=["flac", "copy"], default="flac")
    parser.add_argument("--media-audio-rows-per-shard", type=int, default=256)
    parser.add_argument("--example-rows-per-shard", type=int, default=50_000)
    parser.add_argument("--media-audio-write-batch-size", type=int, default=2)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-examples-per-jsonl", type=int, default=None,
                        help="Debug only: limit examples read from each JSONL.")
    default_workers = min(int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1), 16)
    parser.add_argument("--num-workers", type=int, default=default_workers)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.raw_root.is_dir():
        raise FileNotFoundError(f"raw root not found: {args.raw_root}")
    if args.media_audio_rows_per_shard <= 0 or args.example_rows_per_shard <= 0:
        raise ValueError("rows per shard must be positive")
    if args.media_audio_write_batch_size <= 0:
        raise ValueError("media/audio write batch size must be positive")

    print(f"raw root:   {args.raw_root}", flush=True)
    print(f"output dir: {args.output_dir}", flush=True)

    examples, audio_paths = read_examples(args.raw_root, args.max_examples_per_jsonl)
    splits = {row["split"] for row in examples}
    tasks = {row["task_dir"] for row in examples}
    print(
        f"loaded {len(examples):,} examples across {len(splits)} splits and "
        f"{len(tasks)} task dirs; {len(audio_paths):,} unique referenced audio files",
        flush=True,
    )

    write_source_manifest(args.raw_root, args.output_dir, args)
    example_summary = write_examples(
        examples,
        args.output_dir,
        args.example_rows_per_shard,
        args.compression,
        args.overwrite,
    )
    media_summary = write_media_audio(
        audio_paths,
        args.raw_root,
        args.output_dir,
        args.media_audio_rows_per_shard,
        args.num_workers,
        args.audio_codec,
        args.media_audio_write_batch_size,
        args.overwrite,
    )
    write_manifest(args.output_dir, examples, audio_paths, example_summary, media_summary, args)

    num_media_audio_shards = math.ceil(len(audio_paths) / args.media_audio_rows_per_shard)
    print(
        f"done: {len(examples):,} examples, {len(audio_paths):,} media/audio rows, "
        f"{num_media_audio_shards:,} media/audio shards",
        flush=True,
    )


if __name__ == "__main__":
    main()
