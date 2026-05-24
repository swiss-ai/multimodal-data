#!/usr/bin/env python3

import inspect
import os
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import video2dataset

INPUT_PATH = Path("/tmp/metadata/podcast-transcripts/filtered/metadata.parquet")
DOWNLOAD_INPUT_PATH = INPUT_PATH.with_name("metadata.download_ready.parquet")
OUTPUT_DIR = Path("/path/to/data/audio-datasets/raw(processed)/hf___rchiera___podcast-transcripts")
SAVE_ADDITIONAL_COLUMNS = [
    "episode_id",
    "podcast_slug",
    "episode_slug",
    "episode_title",
    "description",
    "published_at",
    "duration_seconds",
    "belief_count",
    "speaker_slugs",
    "topic_tags",
    "source_url",
]
ENCODE_FORMATS = {"audio": "flac"}
VIDEO2DATASET_CONFIG = {
    "subsampling": {},
    "reading": {
        "yt_args": {
            "download_size": 360,
            "download_audio_rate": 44100,
            "yt_metadata_args": {
                "writesubtitles": "all",
                "subtitleslangs": ["en"],
                "writeautomaticsub": True,
                "get_info": True,
            },
        },
        "timeout": 120,
        "sampler": None,
    },
    "storage": {
        "number_sample_per_shard": 5000,
        "oom_shard_count": 5,
        "captions_are_subtitles": False,
    },
    "distribution": {
        "processes_count": 1,
        "thread_count": 1,
        "subjob_size": 1,
        "distributor": "multiprocessing",
    },
}
BLOCKED_HOSTS = {"example.com", "www.example.com"}


def resolve_video2dataset_callable():
    if hasattr(video2dataset, "download"):
        return video2dataset.download
    if hasattr(video2dataset, "video2dataset"):
        return video2dataset.video2dataset
    raise RuntimeError("Could not find a video2dataset entrypoint.")


def ensure_audio_api(download_fn):
    parameters = set(inspect.signature(download_fn).parameters)
    if "encode_formats" not in parameters:
        raise RuntimeError(
            "The installed video2dataset does not expose the newer audio-capable "
            "`encode_formats` API. This pipeline expects a build that supports "
            "audio downloads."
        )


def normalize_host(host):
    return host.rstrip(".").lower()


def prepare_download_input():
    parquet_file = pq.ParquetFile(INPUT_PATH)
    schema = parquet_file.schema_arrow
    column_names = schema.names

    if DOWNLOAD_INPUT_PATH.exists():
        DOWNLOAD_INPUT_PATH.unlink()

    temp_path = DOWNLOAD_INPUT_PATH.with_suffix(".parquet.tmp")
    if temp_path.exists():
        temp_path.unlink()

    kept = 0
    dropped = 0
    with pq.ParquetWriter(temp_path, schema) as writer:
        for batch in parquet_file.iter_batches(batch_size=1024, columns=column_names):
            data = batch.to_pydict()
            output_rows = {name: [] for name in column_names}

            for index, url in enumerate(data["url"]):
                parts = urlsplit(str(url))
                host = normalize_host(parts.hostname or "")
                if host in BLOCKED_HOSTS:
                    dropped += 1
                    continue

                for name in column_names:
                    output_rows[name].append(data[name][index])
                kept += 1

            if output_rows["url"]:
                writer.write_table(pa.table(output_rows, schema=schema))

    temp_path.rename(DOWNLOAD_INPUT_PATH)
    print(
        f"prepared {DOWNLOAD_INPUT_PATH.name} kept={kept} dropped={dropped}",
        flush=True,
    )


def clear_proxy_env():
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ]:
        os.environ.pop(key, None)


def main():
    if not INPUT_PATH.exists():
        raise RuntimeError(f"Missing filtered parquet: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_download_input()
    clear_proxy_env()

    download_fn = resolve_video2dataset_callable()
    ensure_audio_api(download_fn)

    config = {
        "url_list": str(DOWNLOAD_INPUT_PATH),
        "output_folder": str(OUTPUT_DIR),
        "output_format": "webdataset",
        "input_format": "parquet",
        "encode_formats": ENCODE_FORMATS,
        "stage": "download",
        "url_col": "url",
        "caption_col": "caption",
        "save_additional_columns": SAVE_ADDITIONAL_COLUMNS,
        "enable_wandb": False,
        "incremental_mode": "incremental",
        "max_shard_retry": 1,
        "tmp_dir": "/tmp",
        "config": VIDEO2DATASET_CONFIG,
    }

    supported_parameters = set(inspect.signature(download_fn).parameters)
    filtered_config = {key: value for key, value in config.items() if key in supported_parameters}
    download_fn(**filtered_config)


if __name__ == "__main__":
    main()
