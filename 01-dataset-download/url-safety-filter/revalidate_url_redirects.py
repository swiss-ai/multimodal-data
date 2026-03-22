"""Revalidate live redirect targets for parquet URL datasets before fetching content.

This is intended for datasets such as FaceCaption-15M where the parquet files only
contain URLs and a later stage performs the actual image download. The script
follows live redirects, checks every hop against the same URL safety policy used
by `filter_unsafe_urls.py`, and writes a filtered parquet with the resolved final
URL for rows that remain safe.
"""

from __future__ import annotations

import argparse
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from filter_unsafe_urls import (
    build_blocked_domain_set,
    download_all_blocklists,
    extract_domain,
    load_blocklist,
    resolve_output_path,
    save_blocklist,
    should_block_url,
)


DEFAULT_USER_AGENT = "url-safety-filter/1.0"
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 20.0
DEFAULT_MAX_REDIRECTS = 10
DEFAULT_WORKERS = 64

_THREAD_STATE = threading.local()


def normalize_content_type(raw: str | None) -> str:
    """Return a normalized response content type."""
    if raw is None:
        return ""
    return raw.split(";", 1)[0].strip().lower()


def validate_redirect_chain(url_chain: list[str], blocked_domains: set[str], blocked_cidrs: list) -> tuple[bool, str | None]:
    """Validate every URL in a redirect chain."""
    if not url_chain:
        return False, "empty_redirect_chain"

    for idx, hop_url in enumerate(url_chain, start=1):
        if should_block_url(hop_url, blocked_domains, blocked_cidrs):
            return False, f"unsafe_redirect_hop_{idx}"

    return True, None


def build_redirect_chain(original_url: str, response: requests.Response) -> list[str]:
    """Build a de-duplicated redirect chain including the final URL."""
    chain = [original_url]
    for hop in [*response.history, response]:
        hop_url = getattr(hop, "url", None)
        if hop_url and hop_url != chain[-1]:
            chain.append(hop_url)
    return chain


def get_session(max_redirects: int, user_agent: str) -> requests.Session:
    """Return a thread-local requests session."""
    cache_key = (max_redirects, user_agent)
    session = getattr(_THREAD_STATE, "session", None)
    session_key = getattr(_THREAD_STATE, "session_key", None)
    if session is None or session_key != cache_key:
        session = requests.Session()
        session.max_redirects = max_redirects
        session.headers.update({"User-Agent": user_agent})
        _THREAD_STATE.session = session
        _THREAD_STATE.session_key = cache_key
    return session


def probe_redirect_target(
    url: str | None,
    blocked_domains: set[str],
    blocked_cidrs: list,
    timeout: tuple[float, float],
    require_image_content_type: bool,
    max_redirects: int,
    user_agent: str,
) -> tuple[bool, str | None, str, str | None]:
    """Follow redirects for a single URL and validate the full chain."""
    if url is None:
        return False, None, "", "missing_url"

    if should_block_url(url, blocked_domains, blocked_cidrs):
        return False, None, "", "unsafe_original_url"

    response = None
    try:
        session = get_session(max_redirects=max_redirects, user_agent=user_agent)
        response = session.get(url, allow_redirects=True, stream=True, timeout=timeout)
        redirect_chain = build_redirect_chain(url, response)
        is_safe, reason = validate_redirect_chain(
            redirect_chain, blocked_domains=blocked_domains, blocked_cidrs=blocked_cidrs
        )
        final_url = redirect_chain[-1]
        content_type = normalize_content_type(response.headers.get("Content-Type"))
        if not is_safe:
            return False, final_url, content_type, reason
        if require_image_content_type and not content_type.startswith("image/"):
            label = content_type or "missing"
            return False, final_url, content_type, f"unexpected_content_type ({label})"
        return True, final_url, content_type, None
    except requests.RequestException as exc:
        return False, None, "", f"http_error ({type(exc).__name__}: {exc})"
    finally:
        if response is not None:
            response.close()


def with_or_replace_column(table: pa.Table, column_name: str, values: list[str | None]) -> pa.Table:
    """Append or replace a string column on a pyarrow table."""
    array = pa.array(values, type=pa.string())
    field_index = table.schema.get_field_index(column_name)
    if field_index >= 0:
        return table.set_column(field_index, column_name, array)
    return table.append_column(column_name, array)


def process_parquet_file(args: tuple) -> tuple[str, int, int, dict[str, int]]:
    """Revalidate one parquet file. Returns (rel_path, original_rows, kept_rows, reason_counts)."""
    (
        pq_path,
        input_base,
        output_base,
        explicit_output_path,
        url_column,
        final_url_column,
        content_type_column,
        blocked_domains,
        blocked_cidrs,
        workers,
        connect_timeout,
        read_timeout,
        require_image_content_type,
        max_redirects,
        user_agent,
    ) = args

    rel = pq_path.relative_to(input_base) if input_base else pq_path.name
    out_path = resolve_output_path(pq_path, input_base, output_base, explicit_output_path)

    table = pq.read_table(pq_path)
    urls = table.column(url_column).to_pylist()
    n_original = len(urls)

    keep_mask = [False] * n_original
    final_urls = [None] * n_original
    final_content_types = [None] * n_original
    reason_counts: Counter[str] = Counter()

    timeout = (connect_timeout, read_timeout)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                probe_redirect_target,
                url,
                blocked_domains,
                blocked_cidrs,
                timeout,
                require_image_content_type,
                max_redirects,
                user_agent,
            ): idx
            for idx, url in enumerate(urls)
        }
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            keep, final_url, content_type, reason = future.result()
            keep_mask[idx] = keep
            final_urls[idx] = final_url
            final_content_types[idx] = content_type or None
            if reason:
                reason_counts[reason] += 1
            completed += 1
            if completed % 10000 == 0 or completed == n_original:
                print(f"  {rel}: validated {completed:,}/{n_original:,} URLs")

    table = with_or_replace_column(table, final_url_column, final_urls)
    table = with_or_replace_column(table, content_type_column, final_content_types)
    filtered = table.filter(pa.array(keep_mask, type=pa.bool_()))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, out_path)

    return str(rel), n_original, len(filtered), dict(reason_counts)


def main():
    parser = argparse.ArgumentParser(
        description="Follow live redirects and filter unsafe final URL targets in parquet datasets"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single parquet file")
    group.add_argument("--input-dir", type=Path, help="Directory of parquet files")
    parser.add_argument("--output", type=Path, help="Output parquet (for --input)")
    parser.add_argument("--output-dir", type=Path, help="Output directory (for --input-dir)")
    parser.add_argument("--url-column", default="url", help="Column containing URLs (default: url)")
    parser.add_argument(
        "--final-url-column",
        default="final_url",
        help="Column name for the resolved final URL (default: final_url)",
    )
    parser.add_argument(
        "--content-type-column",
        default="final_content_type",
        help="Column name for the final response content type (default: final_content_type)",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel HTTP workers")
    parser.add_argument("--connect-timeout", type=float, default=DEFAULT_CONNECT_TIMEOUT)
    parser.add_argument("--read-timeout", type=float, default=DEFAULT_READ_TIMEOUT)
    parser.add_argument("--max-redirects", type=int, default=DEFAULT_MAX_REDIRECTS)
    parser.add_argument(
        "--require-image-content-type",
        action="store_true",
        help="Reject URLs whose final response content type does not start with image/",
    )
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    parser.add_argument(
        "--save-blocklist",
        type=Path,
        help="Save merged blocked domains and CIDRs to file",
    )
    parser.add_argument(
        "--load-blocklist",
        type=Path,
        help="Load pre-computed blocked domains and CIDRs from file",
    )
    parser.add_argument(
        "--allow-partial-blocklists",
        action="store_true",
        help="Proceed even if one or more blocklist sources fail to download",
    )
    parser.add_argument(
        "--no-dns",
        action="store_true",
        help="Skip DNS resolution when building the blocklist from source lists",
    )
    args = parser.parse_args()

    if args.input and args.output_dir:
        parser.error("--output-dir may only be used with --input-dir")
    if args.input_dir and args.output:
        parser.error("--output may only be used with --input")

    if args.input:
        parquets = [args.input]
        input_base = args.input.parent
        output_base = None
    else:
        parquets = sorted(args.input_dir.rglob("*.parquet"))
        input_base = args.input_dir
        output_base = args.output_dir
        if not parquets:
            print(f"No parquet files found in {args.input_dir}")
            sys.exit(1)

    print(f"Found {len(parquets)} parquet file(s)")

    if args.load_blocklist:
        print(f"Loading pre-computed blocklist from {args.load_blocklist}")
        blocked_domains, blocked_cidrs = load_blocklist(args.load_blocklist)
        print(f"Loaded {len(blocked_domains):,} blocked domains")
        print(f"Loaded {len(blocked_cidrs):,} blocked CIDRs")
    else:
        domain_blocklist, blocked_cidrs = download_all_blocklists(
            allow_partial=args.allow_partial_blocklists
        )

        print(f"\nExtracting unique domains from {len(parquets)} parquet file(s)...")
        all_domains = set()
        for pq_path in parquets:
            table = pq.read_table(pq_path, columns=[args.url_column])
            for url in table.column(args.url_column).to_pylist():
                if url:
                    domain = extract_domain(url)
                    if domain:
                        all_domains.add(domain)
        print(f"Unique domains: {len(all_domains):,}")

        blocked_domains = build_blocked_domain_set(
            domain_blocklist,
            blocked_cidrs,
            list(all_domains),
            resolve_dns=not args.no_dns,
        )

        if args.save_blocklist:
            args.save_blocklist.parent.mkdir(parents=True, exist_ok=True)
            save_blocklist(args.save_blocklist, blocked_domains, blocked_cidrs)
            print(
                f"Saved {len(blocked_domains):,} blocked domains and "
                f"{len(blocked_cidrs):,} CIDRs to {args.save_blocklist}"
            )

    print(f"\nRevalidating redirect targets with {args.workers} HTTP workers...")

    if args.input and args.output:
        tasks = [
            (
                args.input,
                input_base,
                output_base,
                args.output,
                args.url_column,
                args.final_url_column,
                args.content_type_column,
                blocked_domains,
                blocked_cidrs,
                args.workers,
                args.connect_timeout,
                args.read_timeout,
                args.require_image_content_type,
                args.max_redirects,
                args.user_agent,
            )
        ]
    else:
        tasks = [
            (
                pq_path,
                input_base,
                output_base,
                None,
                args.url_column,
                args.final_url_column,
                args.content_type_column,
                blocked_domains,
                blocked_cidrs,
                args.workers,
                args.connect_timeout,
                args.read_timeout,
                args.require_image_content_type,
                args.max_redirects,
                args.user_agent,
            )
            for pq_path in parquets
        ]

    total_original = 0
    total_kept = 0
    total_reasons: Counter[str] = Counter()

    for task in tasks:
        rel, n_orig, n_kept, reason_counts = process_parquet_file(task)
        total_original += n_orig
        total_kept += n_kept
        total_reasons.update(reason_counts)
        removed = n_orig - n_kept
        print(f"  {rel}: {n_orig:,} -> {n_kept:,} (-{removed:,})")

    total_removed = total_original - total_kept
    pct = (total_removed / total_original * 100) if total_original else 0.0
    print(f"\nDone. {total_original:,} rows -> {total_kept:,} rows ({total_removed:,} removed, {pct:.2f}%)")
    if total_reasons:
        print("Reject reasons:")
        for reason, count in total_reasons.most_common():
            print(f"  {reason}: {count:,}")


if __name__ == "__main__":
    main()
