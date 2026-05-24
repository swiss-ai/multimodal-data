#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_INPUT_PATH = Path("data.parquet")
DEFAULT_OUTPUT_PATH = Path("filtered.parquet")
DEFAULT_ROBOTS_PARQUET = Path("/path/to/data/users/vsabolce/apertus_v1/robotstxt/fineweb_robots_compressed.parquet")
BATCH_SIZE = 10_000


def parse_args():
    parser = ArgumentParser(description="Filter parquet rows using the Recap robots.txt pipeline.")
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input parquet path. Defaults to ./data.parquet.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path. Defaults to ./filtered.parquet.",
    )
    parser.add_argument(
        "--robots-parquet",
        type=Path,
        default=DEFAULT_ROBOTS_PARQUET,
        help=f"Robots parquet path. Defaults to {DEFAULT_ROBOTS_PARQUET}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Rows per batch while filtering. Defaults to {BATCH_SIZE}.",
    )
    parser.add_argument(
        "--url-col",
        default=None,
        help="URL column name. Defaults to auto-detecting image_url or url.",
    )
    return parser.parse_args()


def normalize_host(host):
    return host.rstrip(".").lower()


def normalize_url(url):
    if not url:
        return None

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


def load_robots(robots_parquet, batch_size):
    robots = {}
    parquet_file = pq.ParquetFile(robots_parquet)
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["domain", "protocol", "content"]):
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
        return False

    parts = urlsplit(normalized_url)
    host = normalize_host(parts.hostname or "")
    if not host:
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


def resolve_url_column(schema, requested_column):
    column_names = schema.names
    if requested_column is not None:
        if requested_column not in column_names:
            raise ValueError(f"URL column {requested_column!r} not found. Available columns: {column_names}")
        return requested_column

    for candidate in ("image_url", "url"):
        if candidate in column_names:
            return candidate

    raise ValueError(f"Could not auto-detect a URL column. Expected image_url or url, found: {column_names}")


def filter_parquet(input_path, output_path, robots, batch_size, url_col):
    parquet_file = pq.ParquetFile(input_path)
    schema = parquet_file.schema_arrow
    resolved_url_col = resolve_url_column(schema, url_col)
    url_index = schema.get_field_index(resolved_url_col)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")

    if temp_path.exists():
        temp_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0

    with pq.ParquetWriter(temp_path, schema) as writer:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            urls = batch.column(url_index).to_pylist()
            mask = [allowed_by_robots(url, robots) for url in urls]
            file_kept = sum(mask)
            kept += file_kept
            dropped += len(mask) - file_kept

            if file_kept:
                filtered_batch = pa.Table.from_batches([batch]).filter(pa.array(mask, type=pa.bool_()))
                writer.write_table(filtered_batch)

    temp_path.replace(output_path)
    return resolved_url_col, kept, dropped


def main():
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.input_path}")
    if not args.robots_parquet.exists():
        raise FileNotFoundError(f"Robots parquet not found: {args.robots_parquet}")

    print(f"loading robots from {args.robots_parquet}", flush=True)
    robots = load_robots(args.robots_parquet, args.batch_size)
    url_col, kept, dropped = filter_parquet(
        input_path=args.input_path,
        output_path=args.output_path,
        robots=robots,
        batch_size=args.batch_size,
        url_col=args.url_col,
    )
    print(
        f"filtered {args.input_path} -> {args.output_path} url_col={url_col} kept={kept} dropped={dropped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
