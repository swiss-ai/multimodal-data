#!/usr/bin/env python3

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import webdataset as wds

INPUT_DIR = Path("/path/to/data/audio-datasets/raw(processed)/hf___rchiera___podcast-transcripts")
CHUNKS_PATH = Path("/path/to/data/audio-datasets/raw/hf___rchiera___podcast-transcripts/data/transcript_chunks.parquet")
OUTPUT_DIR = Path("/path/to/data/audio-datasets/raw(processed)/hf___rchiera___podcast-transcripts___chunked")
OUTPUT_METADATA_PATH = OUTPUT_DIR / "metadata.parquet"
OUTPUT_STATS_PATH = OUTPUT_DIR / "chunk_stats.json"
SHARD_PATTERN = str(OUTPUT_DIR / "%05d.tar")
CHUNKS_PER_SHARD = 5_000
MIN_CHUNK_DURATION_SECONDS = 0.05
MAX_WORKERS = 16
MAX_IN_FLIGHT = 64
LOG_EVERY_EPISODES = 50
OUTPUT_AUDIO_EXT = "flac"
OUTPUT_SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("episode_id", pa.string()),
        ("chunk_id", pa.string()),
        ("podcast_slug", pa.string()),
        ("episode_slug", pa.string()),
        ("text", pa.string()),
        ("timestamp_start", pa.float64()),
        ("timestamp_end", pa.float64()),
        ("chunk_index", pa.int64()),
        ("primary_speaker", pa.string()),
        ("speakers", pa.string()),
        ("overlap_tokens", pa.int64()),
        ("parent_key", pa.string()),
        ("audio_ext", pa.string()),
        ("source_url", pa.string()),
    ]
)


def load_chunk_rows():
    table = pq.read_table(CHUNKS_PATH)
    chunks_by_episode = defaultdict(list)
    for row in table.to_pylist():
        chunks_by_episode[row["episode_id"]].append(row)

    for rows in chunks_by_episode.values():
        rows.sort(key=lambda row: row.get("chunk_index") or 0)

    return chunks_by_episode


def infer_audio_ext(sample):
    for ext in ("mp3", "m4a", "wav", "flac"):
        if ext in sample:
            return ext
    return None


def clip_audio(input_path, output_path, start_time, end_time):
    duration = end_time - start_time
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ss",
        f"{start_time:.6f}",
        "-t",
        f"{duration:.6f}",
        "-vn",
        "-map",
        "a:0",
        "-c:a",
        "flac",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)


def build_metadata_row(chunk_key, chunk_row, source_meta, parent_key):
    return {
        "key": chunk_key,
        "episode_id": chunk_row["episode_id"],
        "chunk_id": chunk_row["chunk_id"],
        "podcast_slug": chunk_row["podcast_slug"],
        "episode_slug": chunk_row["episode_slug"],
        "text": chunk_row["text"],
        "timestamp_start": float(chunk_row["timestamp_start"]),
        "timestamp_end": float(chunk_row["timestamp_end"]),
        "chunk_index": int(chunk_row["chunk_index"]),
        "primary_speaker": chunk_row.get("primary_speaker") or "",
        "speakers": json.dumps(chunk_row.get("speakers") or [], ensure_ascii=True),
        "overlap_tokens": int(chunk_row.get("overlap_tokens") or 0),
        "parent_key": parent_key,
        "audio_ext": OUTPUT_AUDIO_EXT,
        "source_url": source_meta.get("url", ""),
    }


def prepare_output_dir():
    if OUTPUT_DIR.exists():
        for path in OUTPUT_DIR.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_episode(sample, chunk_rows):
    source_meta = json.loads(sample["json"].decode("utf-8"))
    episode_id = source_meta["episode_id"]
    parent_key = source_meta["key"]
    audio_ext = infer_audio_ext(sample)
    stats = {
        "episode_id": episode_id,
        "skipped_missing_audio": 0,
        "chunks_written": 0,
        "chunks_skipped_invalid_time": 0,
        "chunks_failed_ffmpeg": 0,
    }
    if source_meta.get("status") != "success" or audio_ext is None:
        stats["skipped_missing_audio"] = 1
        return [], stats

    audio_bytes = sample[audio_ext]
    results = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        input_path = Path(tmp_dir) / f"input.{audio_ext}"
        input_path.write_bytes(audio_bytes)

        for chunk_row in chunk_rows:
            start_time = float(chunk_row["timestamp_start"])
            end_time = float(chunk_row["timestamp_end"])
            duration = end_time - start_time
            if duration <= MIN_CHUNK_DURATION_SECONDS:
                stats["chunks_skipped_invalid_time"] += 1
                continue

            output_path = Path(tmp_dir) / f"{chunk_row['chunk_id']}.{OUTPUT_AUDIO_EXT}"

            try:
                clip_audio(input_path, output_path, start_time, end_time)
            except subprocess.CalledProcessError as exc:
                stats["chunks_failed_ffmpeg"] += 1
                results.append(
                    {
                        "type": "error",
                        "chunk_id": chunk_row["chunk_id"],
                        "error": exc.stderr.decode("utf-8", errors="replace"),
                    }
                )
                continue

            if not output_path.exists() or output_path.stat().st_size == 0:
                stats["chunks_failed_ffmpeg"] += 1
                results.append(
                    {
                        "type": "error",
                        "chunk_id": chunk_row["chunk_id"],
                        "error": "ffmpeg produced no output",
                    }
                )
                continue

            metadata_row = build_metadata_row(
                "",
                chunk_row,
                source_meta,
                parent_key,
            )
            results.append(
                {
                    "type": "sample",
                    "metadata": metadata_row,
                    "text": chunk_row["text"] or "",
                    "json_obj": {
                        **source_meta,
                        **metadata_row,
                        "speakers": chunk_row.get("speakers") or [],
                    },
                    "audio_bytes": output_path.read_bytes(),
                }
            )
            stats["chunks_written"] += 1

    return results, stats


def finalize_result(result, chunk_key):
    metadata = result["metadata"]
    metadata["key"] = chunk_key
    result["json_obj"]["key"] = chunk_key
    return {
        "__key__": chunk_key,
        OUTPUT_AUDIO_EXT: result["audio_bytes"],
        "txt": result["text"].encode("utf-8"),
        "json": json.dumps(result["json_obj"], ensure_ascii=True).encode("utf-8"),
    }, metadata


def flush_done_futures(done, shard_writer, metadata_rows, stats, chunk_counter):
    for future in done:
        results, episode_stats = future.result()
        stats["episodes_seen"] += 1
        if episode_stats["skipped_missing_audio"]:
            stats["episodes_skipped_missing_audio"] += 1
            continue

        stats["episodes_with_chunks"] += 1
        stats["chunks_skipped_invalid_time"] += episode_stats["chunks_skipped_invalid_time"]
        stats["chunks_failed_ffmpeg"] += episode_stats["chunks_failed_ffmpeg"]

        for result in results:
            if result["type"] != "sample":
                continue

            chunk_key = f"{chunk_counter:012d}"
            chunk_counter += 1
            sample, metadata = finalize_result(result, chunk_key)
            shard_writer.write(sample)
            metadata_rows.append(metadata)
            stats["chunks_written"] += 1

    return chunk_counter


def write_outputs(chunks_by_episode):
    shard_writer = wds.ShardWriter(
        SHARD_PATTERN,
        maxcount=CHUNKS_PER_SHARD,
        post=None,
    )
    metadata_rows = []
    stats = {
        "episodes_seen": 0,
        "episodes_with_chunks": 0,
        "episodes_missing_chunks": 0,
        "episodes_skipped_missing_audio": 0,
        "chunks_written": 0,
        "chunks_skipped_invalid_time": 0,
        "chunks_failed_ffmpeg": 0,
    }

    tar_paths = sorted(INPUT_DIR.glob("*.tar"))
    if not tar_paths:
        raise RuntimeError(f"No tar shards found in {INPUT_DIR}")

    chunk_counter = 0
    futures = set()
    submitted_episodes = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for tar_path in tar_paths:
            dataset = wds.WebDataset(str(tar_path), shardshuffle=False)
            for sample in dataset:
                source_meta = json.loads(sample["json"].decode("utf-8"))
                episode_id = source_meta["episode_id"]
                chunk_rows = chunks_by_episode.get(episode_id)
                if not chunk_rows:
                    stats["episodes_seen"] += 1
                    stats["episodes_missing_chunks"] += 1
                    continue

                futures.add(executor.submit(process_episode, sample, chunk_rows))
                submitted_episodes += 1

                if len(futures) >= MAX_IN_FLIGHT:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    chunk_counter = flush_done_futures(
                        done,
                        shard_writer,
                        metadata_rows,
                        stats,
                        chunk_counter,
                    )

                if submitted_episodes % LOG_EVERY_EPISODES == 0:
                    print(
                        f"submitted_episodes={submitted_episodes} "
                        f"episodes_seen={stats['episodes_seen']} "
                        f"chunks_written={stats['chunks_written']}",
                        flush=True,
                    )

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            chunk_counter = flush_done_futures(
                done,
                shard_writer,
                metadata_rows,
                stats,
                chunk_counter,
            )

    shard_writer.close()

    if metadata_rows:
        table = pa.table(
            {name: [row[name] for row in metadata_rows] for name in OUTPUT_SCHEMA.names},
            schema=OUTPUT_SCHEMA,
        )
    else:
        table = pa.table({name: [] for name in OUTPUT_SCHEMA.names}, schema=OUTPUT_SCHEMA)
    pq.write_table(table, OUTPUT_METADATA_PATH)

    OUTPUT_STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


def main():
    if not CHUNKS_PATH.exists():
        raise RuntimeError(f"Missing transcript chunks parquet: {CHUNKS_PATH}")
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Missing downloaded audio directory: {INPUT_DIR}")

    prepare_output_dir()
    chunks_by_episode = load_chunk_rows()
    write_outputs(chunks_by_episode)


if __name__ == "__main__":
    main()
