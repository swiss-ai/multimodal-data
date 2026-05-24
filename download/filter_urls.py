#!/usr/bin/env python3

from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = Path("metadata/parquet")
OUTPUT_DIR = Path("metadata/filtered")
ROBOTS_PARQUET = Path("/path/to/data/users/vsabolce/apertus_v1/robotstxt/fineweb_robots_compressed.parquet")
BLACKLIST_HOSTS = Path("blacklist/hosts")
BLACKLIST_URLS = Path("blacklist/urls_full.txt")
BATCH_SIZE = 100_000
WORKER_COUNT = 50
OUTPUT_SCHEMA = pa.schema([("url", pa.string()), ("caption", pa.string())])
HOST_BLACKLIST = None
URL_BLACKLIST = None
ROBOTS = None


def normalize_host(host):
    return host.rstrip(".").lower()


def normalize_url(url):
    parts = urlsplit(url)
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


def load_blacklists():
    with BLACKLIST_HOSTS.open() as handle:
        host_blacklist = {line.strip().lower() for line in handle if line.strip()}
    with BLACKLIST_URLS.open() as handle:
        url_blacklist = {
            normalized
            for line in handle
            if line.strip()
            for normalized in [normalize_url(line.strip())]
            if normalized is not None
        }
    return host_blacklist, url_blacklist


def load_robots():
    robots = {}
    parquet_file = pq.ParquetFile(ROBOTS_PARQUET)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["domain", "protocol", "content"]):
        data = batch.to_pydict()
        for domain, protocol, content in zip(data["domain"], data["protocol"], data["content"]):
            robots[(protocol.lower(), domain.lower())] = content
    return robots


def is_blacklisted(host, blacklist):
    parts = host.lower().split(".")
    for index in range(len(parts)):
        if ".".join(parts[index:]) in blacklist:
            return True
    return False


@lru_cache(maxsize=200_000)
def parse_robots(content):
    groups = []
    current_agents = []
    current_rules = []

    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue

        field, value = line.split(":", 1)
        field = field.strip().lower()
        value = value.strip()

        if field == "user-agent":
            if current_rules:
                groups.append((current_agents, current_rules))
                current_agents = []
                current_rules = []
            current_agents.append(value.lower())
            continue

        if field in {"allow", "disallow"}:
            current_rules.append((field == "allow", value))

    if current_agents or current_rules:
        groups.append((current_agents, current_rules))

    compiled = []
    for agents, rules in groups:
        if "*" not in agents:
            continue
        for allow, pattern in rules:
            if not pattern:
                continue
            compiled.append((allow, compile_pattern(pattern), len(pattern)))
    return compiled


def compile_pattern(pattern):
    anchored = pattern.endswith("$")
    if anchored:
        pattern = pattern[:-1]
    elif not pattern.endswith("*"):
        pattern = f"{pattern}*"
    return pattern, anchored


def pattern_matches(path, compiled_pattern):
    pattern, anchored = compiled_pattern
    accept_state = len(pattern)
    states = expand_states(pattern, {0})
    if not anchored and accept_state in states:
        return True

    for char in path:
        next_states = set()
        for state in states:
            if state == accept_state:
                continue
            if pattern[state] == "*":
                next_states.add(state)
            elif pattern[state] == char:
                next_states.add(state + 1)

        states = expand_states(pattern, next_states)
        if not anchored and accept_state in states:
            return True

    return accept_state in states


def expand_states(pattern, states):
    expanded = set(states)
    stack = list(states)

    while stack:
        state = stack.pop()
        if state < len(pattern) and pattern[state] == "*" and state + 1 not in expanded:
            expanded.add(state + 1)
            stack.append(state + 1)

    return expanded


def allowed_by_robots(url, host_blacklist, url_blacklist, robots):
    normalized_url = normalize_url(url)
    if normalized_url is None:
        return False

    parts = urlsplit(normalized_url)
    host = normalize_host(parts.hostname or "")
    if not host:
        return False
    if normalized_url in url_blacklist:
        return False
    if is_blacklisted(host, host_blacklist):
        return False

    content = robots.get((parts.scheme.lower(), host))
    if content is None:
        return True

    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"

    best_length = -1
    best_allow = True
    for allow, pattern, length in parse_robots(content):
        matched = pattern_matches(path, pattern)
        if matched and (length > best_length or (length == best_length and allow)):
            best_length = length
            best_allow = allow
    return best_allow


def filter_file(input_path, host_blacklist, url_blacklist, robots):
    output_path = OUTPUT_DIR / input_path.name
    temp_path = OUTPUT_DIR / f"{input_path.name}.tmp"
    kept = 0
    dropped = 0

    if temp_path.exists():
        temp_path.unlink()

    parquet_file = pq.ParquetFile(input_path)
    with pq.ParquetWriter(temp_path, OUTPUT_SCHEMA) as writer:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["url", "caption"]):
            data = batch.to_pydict()
            output_rows = {"url": [], "caption": []}

            for url, caption in zip(data["url"], data["caption"]):
                if allowed_by_robots(url, host_blacklist, url_blacklist, robots):
                    output_rows["url"].append(url)
                    output_rows["caption"].append(caption)
                    kept += 1
                else:
                    dropped += 1

            if output_rows["url"]:
                writer.write_table(pa.table(output_rows, schema=OUTPUT_SCHEMA))

    temp_path.rename(output_path)
    print(f"{input_path.name} kept={kept} dropped={dropped}", flush=True)
    return kept, dropped


def filter_file_worker(input_path):
    return filter_file(input_path, HOST_BLACKLIST, URL_BLACKLIST, ROBOTS)


def main():
    global HOST_BLACKLIST, URL_BLACKLIST, ROBOTS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob("metadata_*.parquet"))
    if not input_files:
        raise RuntimeError(f"No metadata parquet files found in {INPUT_DIR}")

    HOST_BLACKLIST, URL_BLACKLIST = load_blacklists()
    ROBOTS = load_robots()
    kept = 0
    dropped = 0
    pending_files = []

    for input_path in input_files:
        output_path = OUTPUT_DIR / input_path.name
        if output_path.exists():
            print(f"skip {input_path.name}", flush=True)
            continue
        pending_files.append(input_path)

    if not pending_files:
        print("total kept=0 dropped=0", flush=True)
        return

    with get_context("fork").Pool(processes=WORKER_COUNT) as pool:
        for file_kept, file_dropped in pool.imap_unordered(filter_file_worker, pending_files):
            kept += file_kept
            dropped += file_dropped

    print(f"total kept={kept} dropped={dropped}", flush=True)


if __name__ == "__main__":
    main()
