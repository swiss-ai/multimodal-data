#!/usr/bin/env python3

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = Path("/tmp/metadata/HQ-50K/parquet")
OUTPUT_DIR = Path("/tmp/metadata/HQ-50K/filtered")
INPUT_PATH = INPUT_DIR / "metadata.parquet"
OUTPUT_PATH = OUTPUT_DIR / "metadata.parquet"
ROBOTS_PARQUET = Path("/path/to/data/users/vsabolce/apertus_v1/robotstxt/fineweb_robots_compressed.parquet")
BATCH_SIZE = 100_000
OUTPUT_SCHEMA = pa.schema([("url", pa.string())])


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


def load_robots():
    robots = {}
    parquet_file = pq.ParquetFile(ROBOTS_PARQUET)
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["domain", "protocol", "content"]):
        data = batch.to_pydict()
        for domain, protocol, content in zip(data["domain"], data["protocol"], data["content"]):
            robots[(protocol.lower(), domain.lower())] = content
    return robots


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


def allowed_by_robots(url, robots):
    normalized_url = normalize_url(url)
    if normalized_url is None:
        return None

    parts = urlsplit(normalized_url)
    host = normalize_host(parts.hostname or "")
    if not host:
        return None

    content = robots.get((parts.scheme.lower(), host))
    if content is None:
        return normalized_url

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
    if best_allow:
        return normalized_url
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_DIR / f"{OUTPUT_PATH.name}.tmp"

    if OUTPUT_PATH.exists():
        print(f"skip {OUTPUT_PATH.name}", flush=True)
        return

    if temp_path.exists():
        temp_path.unlink()

    if not INPUT_PATH.exists():
        raise RuntimeError(f"Missing metadata parquet: {INPUT_PATH}")

    robots = load_robots()
    kept = 0
    dropped = 0

    parquet_file = pq.ParquetFile(INPUT_PATH)
    with pq.ParquetWriter(temp_path, OUTPUT_SCHEMA) as writer:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=["url"]):
            data = batch.to_pydict()
            output_rows = {"url": []}

            for url in data["url"]:
                filtered_url = allowed_by_robots(url, robots)
                if filtered_url is None:
                    dropped += 1
                    continue
                output_rows["url"].append(filtered_url)
                kept += 1

            if output_rows["url"]:
                writer.write_table(pa.table(output_rows, schema=OUTPUT_SCHEMA))

    temp_path.rename(OUTPUT_PATH)
    print(f"done: kept={kept} dropped={dropped} -> {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
