#!/usr/bin/env python3

import argparse
import csv
import ipaddress
import re
import sqlite3
import sys
from bisect import bisect_right
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

ROOT_DIR = Path("/tmp/metadata/DanQing100M")
INPUT_DIR = ROOT_DIR / "filtered"
OUTPUT_DIR = ROOT_DIR / "verified"
FEEDS_DIR = ROOT_DIR / "cybercrime_feeds"
URLS_DIR = FEEDS_DIR / "urls"
HOSTS_DIR = FEEDS_DIR / "hosts"
IPS_DIR = FEEDS_DIR / "ip"
DB_PATH = FEEDS_DIR / "indicators.sqlite3"

CSV_BATCH_SIZE = 10_000
SQL_BATCH_SIZE = 10_000
URL_PATTERN = re.compile(r"https?://[^\s<>'\"()]+", re.IGNORECASE)
TOKEN_SPLIT_PATTERN = re.compile(r"[\s,]+")


def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build-db", action="store_true")
    group.add_argument("--part-id")
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def set_csv_field_size_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def iter_feed_files(root):
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def iter_clean_lines(path):
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def normalize_url(url):
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    hostname = (parts.hostname or "").lower()
    if not scheme or not hostname:
        return None

    try:
        port = parts.port
    except ValueError:
        return None

    default_port = {"http": 80, "https": 443}.get(scheme)
    netloc = hostname if port in (None, default_port) else f"{hostname}:{port}"
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return f"{scheme}://{netloc}{path}"


def looks_like_hostname(value):
    candidate = value.strip().strip("[]").rstrip(".").lower()
    if not candidate or "/" in candidate or ":" in candidate:
        return False
    if candidate.startswith("http://") or candidate.startswith("https://"):
        return False
    try:
        ipaddress.ip_address(candidate)
    except ValueError:
        pass
    else:
        return False
    return "." in candidate


def iter_url_indicators():
    for path in iter_feed_files(URLS_DIR):
        for line in iter_clean_lines(path):
            matches = URL_PATTERN.findall(line)
            if matches:
                for match in matches:
                    normalized = normalize_url(match)
                    if normalized is not None:
                        yield normalized
                continue

            normalized = normalize_url(line.split()[0])
            if normalized is not None:
                yield normalized


def iter_host_indicators():
    for path in iter_feed_files(HOSTS_DIR):
        for line in iter_clean_lines(path):
            matches = URL_PATTERN.findall(line)
            if matches:
                for match in matches:
                    parts = urlsplit(match)
                    if parts.hostname:
                        yield parts.hostname.lower()
                continue

            tokens = [token.strip().strip("[]") for token in line.split()]
            for token in tokens:
                if looks_like_hostname(token):
                    yield token.rstrip(".").lower()


def iter_ip_networks():
    for path in iter_feed_files(IPS_DIR):
        for line in iter_clean_lines(path):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue

            for token in TOKEN_SPLIT_PATTERN.split(line):
                candidate = token.strip().strip("[]")
                if not candidate:
                    continue
                try:
                    yield ipaddress.ip_network(candidate, strict=False)
                except ValueError:
                    continue


def init_db(connection):
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE urls (
            normalized_url TEXT PRIMARY KEY
        );

        CREATE TABLE hosts (
            host TEXT PRIMARY KEY
        );
        """
    )


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_db(force_rebuild):
    FEEDS_DIR.mkdir(parents=True, exist_ok=True)
    URLS_DIR.mkdir(parents=True, exist_ok=True)
    HOSTS_DIR.mkdir(parents=True, exist_ok=True)
    IPS_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists() and not force_rebuild:
        raise RuntimeError(f"{DB_PATH} already exists. Use --force-rebuild to replace it.")

    url_files = list(iter_feed_files(URLS_DIR))
    host_files = list(iter_feed_files(HOSTS_DIR))
    ip_files = list(iter_feed_files(IPS_DIR))
    if not url_files and not host_files and not ip_files:
        raise RuntimeError(f"No feed files found in {FEEDS_DIR}. Populate urls/, hosts/, or ip/ first.")

    with NamedTemporaryFile(prefix="cybercrime_", suffix=".sqlite3", dir=FEEDS_DIR, delete=False) as tmp_handle:
        tmp_path = Path(tmp_handle.name)

    try:
        connection = sqlite3.connect(tmp_path)
        init_db(connection)

        url_count = 0
        for batch in batched(iter_url_indicators(), SQL_BATCH_SIZE):
            connection.executemany(
                "INSERT OR IGNORE INTO urls(normalized_url) VALUES (?)",
                ((item,) for item in batch),
            )
            url_count += len(batch)

        host_count = 0
        for batch in batched(iter_host_indicators(), SQL_BATCH_SIZE):
            connection.executemany(
                "INSERT OR IGNORE INTO hosts(host) VALUES (?)",
                ((item,) for item in batch),
            )
            host_count += len(batch)

        connection.commit()
        connection.close()
        tmp_path.replace(DB_PATH)

    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    ip_count = sum(1 for _ in iter_ip_networks())
    print(
        f"built {DB_PATH} url_indicators={url_count} host_indicators={host_count} ip_indicators={ip_count}",
        flush=True,
    )


def open_db():
    if not DB_PATH.exists():
        raise RuntimeError(f"Missing {DB_PATH}. Run `python c_verify_urls.py --build-db` first.")

    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def collapse_intervals(intervals):
    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def load_ip_index():
    ipv4 = []
    ipv6 = []

    for network in iter_ip_networks():
        interval = (int(network.network_address), int(network.broadcast_address))
        if network.version == 4:
            ipv4.append(interval)
        else:
            ipv6.append(interval)

    ipv4 = collapse_intervals(ipv4)
    ipv6 = collapse_intervals(ipv6)

    return {
        4: ([start for start, _ in ipv4], [end for _, end in ipv4]),
        6: ([start for start, _ in ipv6], [end for _, end in ipv6]),
    }


def ip_is_blocked(address, ip_index):
    starts, ends = ip_index[address.version]
    if not starts:
        return False

    position = bisect_right(starts, int(address)) - 1
    if position < 0:
        return False
    return int(address) <= ends[position]


def host_suffixes(host):
    parts = host.split(".")
    return [".".join(parts[index:]) for index in range(len(parts))]


def any_host_blocked(connection, host):
    suffixes = host_suffixes(host)
    placeholders = ",".join("?" for _ in suffixes)
    query = f"SELECT 1 FROM hosts WHERE host IN ({placeholders}) LIMIT 1"
    return connection.execute(query, suffixes).fetchone() is not None


def classify_url(url, connection, ip_index):
    normalized = normalize_url(url)
    if normalized is None:
        return None

    if connection.execute("SELECT 1 FROM urls WHERE normalized_url = ? LIMIT 1", (normalized,)).fetchone() is not None:
        return "blocked_url"

    host = (urlsplit(normalized).hostname or "").lower()
    if not host:
        return None

    if any_host_blocked(connection, host):
        return "blocked_host"

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return None

    if ip_is_blocked(address, ip_index):
        return "blocked_ip"
    return None


def filter_file(part_id, connection, ip_index):
    input_path = INPUT_DIR / f"metadata_{part_id}.csv"
    if not input_path.exists():
        print(f"missing {input_path}, skipping", flush=True)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / input_path.name
    temp_path = OUTPUT_DIR / f"{input_path.name}.tmp"

    if output_path.exists():
        print(f"skip {input_path.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    kept = 0
    dropped_url = 0
    dropped_host = 0
    dropped_ip = 0

    with (
        input_path.open(newline="", encoding="utf-8") as src,
        temp_path.open("w", newline="", encoding="utf-8") as dst,
    ):
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=["url", "caption"])
        writer.writeheader()

        buffer = []
        for row in reader:
            reason = classify_url(row["url"], connection, ip_index)
            if reason == "blocked_url":
                dropped_url += 1
                continue
            if reason == "blocked_host":
                dropped_host += 1
                continue
            if reason == "blocked_ip":
                dropped_ip += 1
                continue

            buffer.append({"url": row["url"], "caption": row["caption"]})
            kept += 1

            if len(buffer) == CSV_BATCH_SIZE:
                writer.writerows(buffer)
                buffer.clear()

        if buffer:
            writer.writerows(buffer)

    temp_path.rename(output_path)
    print(
        f"{input_path.name} kept={kept} dropped_url={dropped_url} dropped_host={dropped_host} dropped_ip={dropped_ip}",
        flush=True,
    )


def main():
    args = parse_args()
    set_csv_field_size_limit()

    if args.build_db:
        build_db(args.force_rebuild)
        return

    connection = open_db()
    try:
        ip_index = load_ip_index()
        filter_file(f"{int(args.part_id):04d}", connection, ip_index)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
