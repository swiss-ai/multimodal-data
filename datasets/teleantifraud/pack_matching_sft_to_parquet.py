#!/usr/bin/env python3
"""Pack TeleAntiFraud's HF-matching SFT split into self-contained Parquet."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from mutagen import File as MutagenFile


RAW_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/raw/"
    "ms___JimmyMa99___TeleAntiFraud-28k/TeleAntiFraud-28k"
)
OUT_ROOT = Path(
    "/capstor/store/cscs/swissai/infra01/audio-datasets/processed/sft/"
    "ms___JimmyMa99___TeleAntiFraud-28k/matching_sft"
)

DATASET_ID = "ms___JimmyMa99___TeleAntiFraud-28k"
VARIANT = "matching_hf_sft_wpd"
SOURCE_FILES = {"train": "total_train_wpd.jsonl", "test": "total_test_wpd.jsonl"}


def normalize_audio_path(path: str) -> str:
    prefix = "/root/code/ChatTTS/"
    if path.startswith(prefix):
        path = path[len(prefix):]
    return path.lstrip("/")


def audio_id(path: str) -> str:
    return "teleantifraud__" + Path(path).with_suffix("").as_posix().replace("/", "__")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mp3_info(path: Path) -> tuple[float, int, int]:
    media = MutagenFile(path)
    if media is None or not getattr(media, "info", None):
        raise RuntimeError(f"could not read mp3 header: {path}")
    info = media.info
    return (
        float(getattr(info, "length", 0.0) or 0.0),
        int(getattr(info, "sample_rate", 0) or 0),
        int(getattr(info, "channels", 0) or 0),
    )


def read_examples(raw_root: Path, max_rows_per_split: int | None) -> tuple[list[dict], list[str]]:
    examples: list[dict] = []
    audio_paths: set[str] = set()

    for split, filename in SOURCE_FILES.items():
        jsonl = raw_root / filename
        with jsonl.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                if max_rows_per_split is not None and line_idx > max_rows_per_split:
                    break
                row = json.loads(line)
                audios = [normalize_audio_path(x) for x in row.get("audios", [])]
                if len(audios) != 1:
                    raise ValueError(f"{jsonl}:{line_idx} expected exactly one audio, got {len(audios)}")
                rel_audio = audios[0]
                aid = audio_id(rel_audio)
                audio_paths.add(rel_audio)
                messages = row.get("messages", [])
                examples.append({
                    "sample_id": f"teleantifraud__{split}__{line_idx:08d}",
                    "dataset_id": DATASET_ID,
                    "variant": VARIANT,
                    "split": split,
                    "messages_json": json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
                    "audio_ids": [aid],
                    "source_audio_path": rel_audio,
                    "source_jsonl": filename,
                    "source_line": line_idx,
                    "turn_count": len(messages),
                })

    missing = [p for p in sorted(audio_paths) if not (raw_root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced audio files are missing; first={missing[0]}")
    return examples, sorted(audio_paths)


def write_examples(examples: list[dict], out_root: Path, compression: str) -> dict:
    out_dir = out_root / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    for split in SOURCE_FILES:
        rows = [row for row in examples if row["split"] == split]
        counts[split] = len(rows)
        path = out_dir / f"{split}.parquet"
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path, compression=compression)
        print(f"wrote {path} ({len(rows):,} rows)", flush=True)
    return {"splits": dict(counts), "shards": len(SOURCE_FILES)}


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def pack_audio_shard(task: tuple[int, int, list[str], str, str, str]) -> dict:
    shard_idx, num_shards, rel_paths, raw_root_s, out_root_s, compression = task
    raw_root = Path(raw_root_s)
    out_root = Path(out_root_s)
    out_dir = out_root / "media" / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"audio-{shard_idx:05d}-of-{num_shards:05d}.parquet"
    shard_path = out_path.relative_to(out_root).as_posix()
    index_rows: list[dict] = []
    failures: list[dict] = []
    total_bytes = 0
    rows: list[dict] = []

    for rel_path in rel_paths:
        path = raw_root / rel_path
        try:
            data = path.read_bytes()
            duration, sample_rate, channels = mp3_info(path)
            digest = sha256(data)
        except Exception as exc:
            failures.append({
                "source_path": rel_path,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            })
            continue

        aid = audio_id(rel_path)
        row = {
            "audio_id": aid,
            "audio_bytes": data,
            "source_path": rel_path,
            "sha256": digest,
            "size_bytes": len(data),
            "format": "mp3",
            "codec": "mp3",
            "duration_sec": duration,
            "sample_rate": sample_rate,
            "channels": channels,
        }
        rows.append(row)
        index_rows.append({
            "audio_id": aid,
            "shard_path": shard_path,
            "row_index": len(rows) - 1,
            "source_path": rel_path,
            "sha256": digest,
            "size_bytes": len(data),
            "duration_sec": duration,
            "sample_rate": sample_rate,
            "channels": channels,
        })
        total_bytes += len(data)

    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_path, compression=compression)

    return {
        "shard_idx": shard_idx,
        "written": len(rows),
        "bytes": total_bytes,
        "index_rows": index_rows,
        "failures": failures,
    }


def write_audio(
    audio_paths: list[str],
    raw_root: Path,
    out_root: Path,
    rows_per_shard: int,
    workers: int,
    compression: str,
) -> dict:
    shards = chunks(audio_paths, rows_per_shard)
    tasks = [
        (idx, len(shards), shard, str(raw_root), str(out_root), compression)
        for idx, shard in enumerate(shards)
    ]
    index_rows: list[dict] = []
    failures: list[dict] = []
    total_written = 0
    total_bytes = 0
    started = time.time()

    print(f"packing {len(audio_paths):,} referenced MP3s into {len(shards):,} shards", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        for future in as_completed([pool.submit(pack_audio_shard, task) for task in tasks]):
            result = future.result()
            total_written += result["written"]
            total_bytes += result["bytes"]
            index_rows.extend(result["index_rows"])
            failures.extend(result["failures"])
            elapsed = (time.time() - started) / 60
            print(
                f"audio shard {result['shard_idx']:05d}: {result['written']:,} rows; "
                f"total={total_written:,}; elapsed={elapsed:.1f}m",
                flush=True,
            )

    out_dir = out_root / "media" / "audio"
    if failures:
        fail_path = out_dir / "_failures.jsonl"
        with fail_path.open("w", encoding="utf-8") as f:
            for row in failures:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        raise RuntimeError(f"{len(failures)} audio files failed; see {fail_path}")

    index_rows.sort(key=lambda row: row["audio_id"])
    pq.write_table(
        pa.Table.from_pylist(index_rows),
        out_dir / "_index.parquet",
        compression=compression,
    )
    return {"rows": total_written, "shards": len(shards), "bytes": total_bytes}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--audio-rows-per-shard", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=min(os.cpu_count() or 1, 64))
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    examples, audio_paths = read_examples(args.raw_root, args.max_rows_per_split)
    example_summary = write_examples(examples, args.output_dir, args.compression)
    audio_summary = write_audio(
        audio_paths,
        args.raw_root,
        args.output_dir,
        args.audio_rows_per_shard,
        args.num_workers,
        args.compression,
    )

    turn_counts = Counter(str(row["turn_count"]) for row in examples)
    write_json(args.output_dir / "_SOURCE.json", {
        "dataset_id": DATASET_ID,
        "variant": VARIANT,
        "source_repo": "modelscope://JimmyMa99/TeleAntiFraud-28k",
        "source_files": SOURCE_FILES,
        "raw_root": str(args.raw_root),
        "source_zip_sha256": "76aabead2c2282ba3f5554836050dfad2111b101f7247593d479cb6de6f4ddf6",
        "source_zip_size_bytes": 13335421925,
        "modelscope_revisions": {
            "metadata": "96af94555e832069e9b26dff69f587f84550b02f",
            "TeleAntiFraud-28k.zip": "49a35978283a6a3b1a358c0dbde71adfdfa45cc2",
        },
        "matches_huggingface": {
            "repo": "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud",
            "revision": "0872e54b584b28d34e0911dffcf696f0b2e5e49a",
        },
        "audio_policy": "copy original MP3 bytes",
        "packed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    write_json(args.output_dir / "_MANIFEST.json", {
        "dataset_id": DATASET_ID,
        "variant": VARIANT,
        "format": "audio_sft_parquet_v1",
        "layout": {
            "examples": "examples/{train,test}.parquet",
            "media_audio": "media/audio/audio-*.parquet",
            "media_audio_index": "media/audio/_index.parquet",
        },
        "num_examples": len(examples),
        "num_unique_audio": len(audio_paths),
        "splits": example_summary["splits"],
        "message_count_distribution": dict(sorted(turn_counts.items())),
        "audio": {
            "codec": "mp3",
            "size_bytes": audio_summary["bytes"],
        },
        "shards": {
            "examples": example_summary["shards"],
            "media_audio": audio_summary["shards"],
        },
        "invariants": {
            "missing_referenced_audio": 0,
            "media_audio_ids_unreferenced_by_examples": 0,
        },
    })
    print(f"done: {len(examples):,} examples, {len(audio_paths):,} audio files", flush=True)


if __name__ == "__main__":
    main()
