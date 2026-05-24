#!/usr/bin/env python3

import json
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_PARQUET = Path("/path/to/data/vision-datasets/raw/OpenFace-CQUPT___FaceCaption-15M/FaceCaption-v2.parquet")
OUTPUT_DIR = Path("/tmp/metadata/FaceCaption-15M/filtered")
ROBOTS_PARQUET = Path("/path/to/data/users/vsabolce/apertus_v1/robotstxt/fineweb_robots_compressed.parquet")
BATCH_SIZE = 100_000
WORKER_COUNT = 15
OUTPUT_SCHEMA = pa.schema(
    [
        ("_id", pa.string()),
        ("caption", pa.string()),
        ("url", pa.string()),
        ("laion_caption", pa.string()),
        ("box", pa.string()),
    ]
)
ROBOTS = None


def part_id_for_row_group(row_group_index):
    return f"{row_group_index:05d}"


def output_path_for_row_group(row_group_index):
    return OUTPUT_DIR / f"metadata_{part_id_for_row_group(row_group_index)}.parquet"


def normalize_host(host):
    return host.rstrip(".").lower()


def extract_source_url(raw_url):
    if raw_url is None:
        return None

    source_url = raw_url.split(", ", 1)[0].strip()
    if not source_url or source_url == "UNKNOWN":
        return None
    return source_url


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


def normalize_box(raw_box):
    if raw_box is None:
        return None

    try:
        box = json.loads(raw_box)
    except json.JSONDecodeError:
        return None

    if not isinstance(box, list) or len(box) != 4:
        return None

    coords = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        coord = int(value)
        if coord != value:
            return None
        coords.append(coord)

    x1, y1, x2, y2 = coords
    if x1 >= x2 or y1 >= y2:
        return None

    return json.dumps(coords)


def require_single_text(field_name, row_id, values):
    if values is None or len(values) != 1 or values[0] is None:
        raise RuntimeError(f"{field_name} for _id={row_id} is not a single string: {values!r}")
    return values[0]


def optional_single_text(field_name, row_id, values):
    if values is None or len(values) == 0:
        return ""
    if len(values) == 1 and values[0] is None:
        return ""
    if len(values) != 1:
        raise RuntimeError(f"{field_name} for _id={row_id} is not a single string: {values!r}")
    return values[0]


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


def allowed_by_robots(normalized_url, robots):
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


def filter_row_group(row_group_index, robots):
    output_path = output_path_for_row_group(row_group_index)
    temp_path = OUTPUT_DIR / f"{output_path.name}.tmp"
    kept = 0
    dropped_bad_url = 0
    dropped_robots = 0
    dropped_bad_box = 0

    if temp_path.exists():
        temp_path.unlink()

    parquet_file = pq.ParquetFile(SOURCE_PARQUET)
    with pq.ParquetWriter(temp_path, OUTPUT_SCHEMA) as writer:
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE,
            row_groups=[row_group_index],
            columns=["_id", "caption", "url", "laion_caption", "box"],
        ):
            data = batch.to_pydict()
            output_rows = {
                "_id": [],
                "caption": [],
                "url": [],
                "laion_caption": [],
                "box": [],
            }

            for row_id, caption_values, raw_url, laion_values, raw_box in zip(
                data["_id"],
                data["caption"],
                data["url"],
                data["laion_caption"],
                data["box"],
            ):
                caption = require_single_text("caption", row_id, caption_values)
                laion_caption = optional_single_text("laion_caption", row_id, laion_values)

                source_url = extract_source_url(raw_url)
                normalized_url = normalize_url(source_url) if source_url else None
                if normalized_url is None:
                    dropped_bad_url += 1
                    continue

                if not allowed_by_robots(normalized_url, robots):
                    dropped_robots += 1
                    continue

                box = normalize_box(raw_box)
                if box is None:
                    dropped_bad_box += 1
                    continue

                output_rows["_id"].append(row_id)
                output_rows["caption"].append(caption)
                output_rows["url"].append(normalized_url)
                output_rows["laion_caption"].append(laion_caption)
                output_rows["box"].append(box)
                kept += 1

            if output_rows["_id"]:
                writer.write_table(pa.table(output_rows, schema=OUTPUT_SCHEMA))

    temp_path.rename(output_path)
    print(
        f"{output_path.name} kept={kept} dropped_bad_url={dropped_bad_url} "
        f"dropped_robots={dropped_robots} dropped_bad_box={dropped_bad_box}",
        flush=True,
    )
    return kept, dropped_bad_url, dropped_robots, dropped_bad_box


def filter_row_group_worker(row_group_index):
    return filter_row_group(row_group_index, ROBOTS)


def main():
    global ROBOTS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_file = pq.ParquetFile(SOURCE_PARQUET)
    row_group_count = parquet_file.metadata.num_row_groups

    ROBOTS = load_robots()
    kept = 0
    dropped_bad_url = 0
    dropped_robots = 0
    dropped_bad_box = 0
    pending_row_groups = []

    for row_group_index in range(row_group_count):
        output_path = output_path_for_row_group(row_group_index)
        if output_path.exists():
            print(f"skip {output_path.name}", flush=True)
            continue
        pending_row_groups.append(row_group_index)

    if not pending_row_groups:
        print(
            "total kept=0 dropped_bad_url=0 dropped_robots=0 dropped_bad_box=0",
            flush=True,
        )
        return

    with get_context("fork").Pool(processes=WORKER_COUNT) as pool:
        for (
            file_kept,
            file_dropped_bad_url,
            file_dropped_robots,
            file_dropped_bad_box,
        ) in pool.imap_unordered(filter_row_group_worker, pending_row_groups):
            kept += file_kept
            dropped_bad_url += file_dropped_bad_url
            dropped_robots += file_dropped_robots
            dropped_bad_box += file_dropped_bad_box

    print(
        f"total kept={kept} dropped_bad_url={dropped_bad_url} "
        f"dropped_robots={dropped_robots} dropped_bad_box={dropped_bad_box}",
        flush=True,
    )


if __name__ == "__main__":
    main()
