#!/usr/bin/env python3

from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq

INPUT_DIR = Path("/tmp/metadata/DanQing100M/filtered")
OUTPUT_FILE = Path("/tmp/metadata/DanQing100M/ba_hosts.txt")
BATCH_SIZE = 100_000


def normalize_host(host):
    return host.rstrip(".").lower()


def extract_host(url):
    parts = urlsplit(url)
    if parts.scheme.lower() not in {"http", "https"}:
        return None

    host = normalize_host(parts.hostname or "")
    if not host:
        return None
    return host


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob("metadata_*.parquet"))
    if not input_files:
        raise RuntimeError(f"No filtered parquet files found in {INPUT_DIR}")

    hosts = set()
    for input_path in input_files:
        parquet_file = pq.ParquetFile(input_path)
        rows = 0
        before = len(hosts)

        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["url"]):
            urls = batch.column("url").to_pylist()
            rows += len(urls)
            for url in urls:
                host = extract_host(url)
                if host is not None:
                    hosts.add(host)

        print(
            f"{input_path.name} rows={rows} new_hosts={len(hosts) - before} total_hosts={len(hosts)}",
            flush=True,
        )

    temp_path = OUTPUT_FILE.with_suffix(".txt.tmp")
    with temp_path.open("w") as handle:
        for host in sorted(hosts):
            handle.write(f"{host}\n")
    temp_path.rename(OUTPUT_FILE)
    print(f"wrote {OUTPUT_FILE} hosts={len(hosts)}", flush=True)


if __name__ == "__main__":
    main()
