#!/usr/bin/env python3

from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = Path("/tmp/metadata/podcast-transcripts/parquet")
OUTPUT_DIR = Path("/tmp/metadata/podcast-transcripts/filtered")
INPUT_PATH = INPUT_DIR / "metadata.parquet"
OUTPUT_PATH = OUTPUT_DIR / "metadata.parquet"
BATCH_SIZE = 1024


def normalize_host(host):
    return host.rstrip(".").lower()


def normalize_url(url):
    if url is None:
        return None

    raw_url = str(url).strip()
    if not raw_url:
        return None

    parts = urlsplit(raw_url)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"}:
        return None

    host = normalize_host(parts.hostname or "")
    if not host:
        return None

    port = parts.port
    if port is not None and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    path = parts.path or "/"
    if parts.query:
        return f"{scheme}://{netloc}{path}?{parts.query}"
    return f"{scheme}://{netloc}{path}"


def filter_file():
    if not INPUT_PATH.exists():
        raise RuntimeError(f"Missing metadata parquet: {INPUT_PATH}")

    temp_path = OUTPUT_DIR / f"{OUTPUT_PATH.name}.tmp"
    if OUTPUT_PATH.exists():
        print(f"skip {OUTPUT_PATH.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    parquet_file = pq.ParquetFile(INPUT_PATH)
    output_schema = parquet_file.schema_arrow
    column_names = output_schema.names
    seen_episode_ids = set()
    seen_urls = set()
    kept = 0
    dropped = 0

    with pq.ParquetWriter(temp_path, output_schema) as writer:
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            columns=column_names,
        ):
            data = batch.to_pydict()
            output_rows = {name: [] for name in column_names}

            for index in range(len(data["episode_id"])):
                episode_id_value = data["episode_id"][index]
                episode_id = "" if episode_id_value is None else str(episode_id_value).strip()
                normalized_url = normalize_url(data["url"][index])

                if (
                    not episode_id
                    or normalized_url is None
                    or episode_id in seen_episode_ids
                    or normalized_url in seen_urls
                ):
                    dropped += 1
                    continue

                seen_episode_ids.add(episode_id)
                seen_urls.add(normalized_url)

                for name in column_names:
                    value = data[name][index]
                    if name == "url":
                        value = normalized_url
                    output_rows[name].append(value)

                kept += 1

            if output_rows["episode_id"]:
                writer.write_table(pa.table(output_rows, schema=output_schema))

    temp_path.rename(OUTPUT_PATH)
    print(f"{INPUT_PATH.name} kept={kept} dropped={dropped}", flush=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filter_file()


if __name__ == "__main__":
    main()
