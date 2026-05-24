#!/usr/bin/env python3

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATASET_DIR = Path("/path/to/data/audio-datasets/raw/hf___rchiera___podcast-transcripts/data")
EPISODE_METADATA_PATH = DATASET_DIR / "episode_metadata.parquet"
TRANSCRIPTS_PATH = DATASET_DIR / "transcripts.parquet"
OUTPUT_DIR = Path("/tmp/metadata/podcast-transcripts/parquet")
OUTPUT_PATH = OUTPUT_DIR / "metadata.parquet"
BATCH_SIZE = 1024
OUTPUT_SCHEMA = pa.schema(
    [
        ("url", pa.string()),
        ("caption", pa.string()),
        ("episode_id", pa.string()),
        ("podcast_slug", pa.string()),
        ("episode_slug", pa.string()),
        ("episode_title", pa.string()),
        ("description", pa.string()),
        ("published_at", pa.string()),
        ("duration_seconds", pa.int64()),
        ("belief_count", pa.int64()),
        ("speaker_slugs", pa.string()),
        ("topic_tags", pa.string()),
        ("source_url", pa.string()),
    ]
)


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_url(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_int(value):
    if value is None:
        return None
    return int(value)


def normalize_json_list(value):
    if value is None:
        return "[]"
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value if item is not None]
    else:
        items = [str(value)]
    return json.dumps(items, ensure_ascii=True)


def load_transcripts():
    transcripts = {}
    parquet_file = pq.ParquetFile(TRANSCRIPTS_PATH)
    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["episode_id", "transcript_text"],
    ):
        data = batch.to_pydict()
        for episode_id, transcript_text in zip(data["episode_id"], data["transcript_text"]):
            normalized_episode_id = normalize_text(episode_id)
            if not normalized_episode_id:
                continue
            transcripts[normalized_episode_id] = normalize_text(transcript_text)
    return transcripts


def iter_metadata_tables(transcripts_by_episode):
    seen_episode_ids = set()
    parquet_file = pq.ParquetFile(EPISODE_METADATA_PATH)

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "episode_id",
            "podcast_slug",
            "episode_slug",
            "title",
            "description",
            "published_at",
            "duration_seconds",
            "belief_count",
            "speaker_slugs",
            "topic_tags",
            "audio_url",
            "source_url",
        ],
    ):
        data = batch.to_pydict()
        rows = {name: [] for name in OUTPUT_SCHEMA.names}

        for index in range(len(data["episode_id"])):
            episode_id = normalize_text(data["episode_id"][index])
            if not episode_id or episode_id in seen_episode_ids:
                continue

            seen_episode_ids.add(episode_id)

            rows["url"].append(normalize_url(data["audio_url"][index]))
            rows["caption"].append(transcripts_by_episode.get(episode_id, ""))
            rows["episode_id"].append(episode_id)
            rows["podcast_slug"].append(normalize_text(data["podcast_slug"][index]))
            rows["episode_slug"].append(normalize_text(data["episode_slug"][index]))
            rows["episode_title"].append(normalize_text(data["title"][index]))
            rows["description"].append(normalize_text(data["description"][index]))
            rows["published_at"].append(normalize_text(data["published_at"][index]))
            rows["duration_seconds"].append(normalize_int(data["duration_seconds"][index]))
            rows["belief_count"].append(normalize_int(data["belief_count"][index]))
            rows["speaker_slugs"].append(normalize_json_list(data["speaker_slugs"][index]))
            rows["topic_tags"].append(normalize_json_list(data["topic_tags"][index]))
            rows["source_url"].append(normalize_text(data["source_url"][index]))

        if rows["episode_id"]:
            yield pa.table(rows, schema=OUTPUT_SCHEMA)


def write_metadata():
    if not EPISODE_METADATA_PATH.exists():
        raise RuntimeError(f"Missing episode metadata parquet: {EPISODE_METADATA_PATH}")
    if not TRANSCRIPTS_PATH.exists():
        raise RuntimeError(f"Missing transcripts parquet: {TRANSCRIPTS_PATH}")

    temp_path = OUTPUT_DIR / f"{OUTPUT_PATH.name}.tmp"

    if OUTPUT_PATH.exists():
        print(f"skip {OUTPUT_PATH.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    transcripts_by_episode = load_transcripts()
    rows_written = 0
    with pq.ParquetWriter(temp_path, OUTPUT_SCHEMA) as writer:
        for table in iter_metadata_tables(transcripts_by_episode):
            writer.write_table(table)
            rows_written += table.num_rows

    temp_path.rename(OUTPUT_PATH)
    print(f"{OUTPUT_PATH.name} rows={rows_written}", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_metadata()


if __name__ == "__main__":
    main()
