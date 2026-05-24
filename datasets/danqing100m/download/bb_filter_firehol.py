#!/usr/bin/env python3

import ipaddress
import socket
import time
from bisect import bisect_right
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

ROOT_DIR = Path("/tmp/metadata/DanQing100M")
INPUT_DIR = ROOT_DIR / "filtered"
HOSTS_FILE = ROOT_DIR / "ba_hosts.txt"
BLOCKED_HOSTS_FILE = ROOT_DIR / "bb_blocked_hosts.txt"
OUTPUT_DIR = ROOT_DIR / "bb_filtered"
FIREHOL_FILE = Path("/tmp/blacklist/firehol_level1.netset")
BATCH_SIZE = 100_000
RESOLVE_WORKERS = 256
FILTER_WORKERS = 16
OUTPUT_SCHEMA = pa.schema([("url", pa.string()), ("caption", pa.string())])
FIREHOL_RANGE_STARTS = None
FIREHOL_RANGE_ENDS = None
BLOCKED_HOSTS = None


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


def load_firehol_ranges():
    starts = []
    ends = []

    with FIREHOL_FILE.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            network = ipaddress.ip_network(line, strict=False)
            if network.version != 4:
                continue
            starts.append(int(network.network_address))
            ends.append(int(network.broadcast_address))

    if not starts:
        raise RuntimeError(f"No IPv4 networks found in {FIREHOL_FILE}")

    paired = sorted(zip(starts, ends))
    sorted_starts, sorted_ends = zip(*paired)
    return tuple(sorted_starts), tuple(sorted_ends)


def ip_in_firehol(ip_obj):
    if ip_obj.version != 4:
        return False

    ip_int = int(ip_obj)
    index = bisect_right(FIREHOL_RANGE_STARTS, ip_int) - 1
    if index < 0:
        return False
    return ip_int <= FIREHOL_RANGE_ENDS[index]


def resolve_host(host):
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addrinfo = socket.getaddrinfo(
                host,
                80,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except (socket.gaierror, UnicodeError):
            return host, False

        addresses = []
        for entry in addrinfo:
            sockaddr = entry[4]
            if not sockaddr:
                continue
            try:
                addresses.append(ipaddress.ip_address(sockaddr[0]))
            except ValueError:
                continue

        if not addresses:
            return host, False

    blocked = any(ip_in_firehol(ip_obj) for ip_obj in set(addresses))
    return host, blocked


def read_hosts():
    with HOSTS_FILE.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def resolve_blocked_hosts(hosts):
    blocked_hosts = set()
    total = len(hosts)
    if not total:
        return blocked_hosts

    started_at = time.monotonic()
    completed = 0
    host_iter = iter(hosts)

    with ThreadPoolExecutor(max_workers=RESOLVE_WORKERS) as executor:
        futures = {}
        initial = min(total, RESOLVE_WORKERS * 4)
        for _ in range(initial):
            host = next(host_iter)
            futures[executor.submit(resolve_host, host)] = host

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                host = futures.pop(future)
                resolved_host, blocked = future.result()
                if blocked:
                    blocked_hosts.add(resolved_host)

                completed += 1
                if completed % 1000 == 0 or completed == total:
                    elapsed = max(time.monotonic() - started_at, 1e-9)
                    rate = completed / elapsed
                    print(
                        f"resolve progress hosts={completed}/{total} blocked={len(blocked_hosts)} "
                        f"hosts_per_sec={rate:.0f}",
                        flush=True,
                    )

                try:
                    next_host = next(host_iter)
                except StopIteration:
                    continue
                futures[executor.submit(resolve_host, next_host)] = next_host

    return blocked_hosts


def write_blocked_hosts(blocked_hosts):
    temp_path = BLOCKED_HOSTS_FILE.with_suffix(".txt.tmp")
    with temp_path.open("w") as handle:
        for host in sorted(blocked_hosts):
            handle.write(f"{host}\n")
    temp_path.rename(BLOCKED_HOSTS_FILE)
    print(f"wrote {BLOCKED_HOSTS_FILE} hosts={len(blocked_hosts)}", flush=True)


def load_blocked_hosts():
    with BLOCKED_HOSTS_FILE.open() as handle:
        return {line.strip() for line in handle if line.strip()}


def filter_file(input_path):
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
                host = extract_host(url)
                if host is None or host in BLOCKED_HOSTS:
                    dropped += 1
                    continue

                output_rows["url"].append(url)
                output_rows["caption"].append(caption)
                kept += 1

            if output_rows["url"]:
                writer.write_table(pa.table(output_rows, schema=OUTPUT_SCHEMA))

    temp_path.rename(output_path)
    print(f"{input_path.name} kept={kept} dropped={dropped}", flush=True)
    return kept, dropped


def filter_file_worker(input_path):
    return filter_file(input_path)


def main():
    global FIREHOL_RANGE_STARTS, FIREHOL_RANGE_ENDS, BLOCKED_HOSTS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob("metadata_*.parquet"))
    if not input_files:
        raise RuntimeError(f"No filtered parquet files found in {INPUT_DIR}")
    if not HOSTS_FILE.exists():
        raise RuntimeError(f"Missing hosts file {HOSTS_FILE}; run ba_gather_targets.py first")

    FIREHOL_RANGE_STARTS, FIREHOL_RANGE_ENDS = load_firehol_ranges()
    hosts = read_hosts()
    print(f"resolving hosts={len(hosts)}", flush=True)
    write_blocked_hosts(resolve_blocked_hosts(hosts))
    BLOCKED_HOSTS = load_blocked_hosts()

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

    with get_context("fork").Pool(processes=FILTER_WORKERS) as pool:
        for file_kept, file_dropped in pool.imap_unordered(filter_file_worker, pending_files):
            kept += file_kept
            dropped += file_dropped

    print(f"total kept={kept} dropped={dropped}", flush=True)


if __name__ == "__main__":
    main()
