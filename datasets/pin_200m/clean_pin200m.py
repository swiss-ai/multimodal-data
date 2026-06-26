#!/usr/bin/env python3
"""Clean PIN-200M manifest using quality scores + robots.txt filtering.

Two-phase approach:
  1. File-parallel: read JSONL docs, extract quality signals + URLs
  2. Single-process: check URLs against robots.txt (loaded once, cached)

Usage::

    python clean_pin200m.py \
        --manifest /path/to/manifest.parquet \
        --output /path/to/manifest_clean.parquet \
        --robots-parquet /path/to/robots.parquet \
        --num-workers 32
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def _check_quality(
    qs: dict,
    *,
    min_text_blocks: int,
    min_avg_tokens_per_block: float,
    min_total_tokens: int,
) -> Optional[str]:
    """Return rejection reason or None."""
    if qs.get("text_block_count", 0) < min_text_blocks:
        return "low_text_blocks"
    if qs.get("avg_tokens_per_text_block", 0) < min_avg_tokens_per_block:
        return "low_avg_tokens"
    if qs.get("total_token_count", 0) < min_total_tokens:
        return "low_total_tokens"
    return None


def _extract_url(doc: dict) -> str:
    """Extract document URL from PIN-200M metadata."""
    meta = doc.get("meta", {})
    if not isinstance(meta, dict):
        return ""
    ori_meta = meta.get("ori_meta")
    if not isinstance(ori_meta, dict):
        return ""
    return (
        ori_meta.get("document_url")
        or ori_meta.get("general_meta", {}).get("url", "")
    )


# ---------------------------------------------------------------------------
# Phase 1: file-parallel extraction
# ---------------------------------------------------------------------------

# Result per doc: (group_id, url, quality_rejection_reason_or_None)
DocResult = Tuple[int, str, Optional[str]]


def _process_file(
    jsonl_path: str,
    docs: List[Tuple[int, int, int]],  # (line_start, line_length, group_id)
    quality_kwargs: dict,
) -> List[DocResult]:
    """Read one JSONL file, extract quality + URL for each doc."""
    results: List[DocResult] = []
    docs_sorted = sorted(docs, key=lambda x: x[0])

    with open(jsonl_path, "rb") as f:
        for line_start, line_length, group_id in docs_sorted:
            try:
                f.seek(line_start)
                doc = orjson.loads(f.read(line_length))
            except Exception:
                results.append((group_id, "", "read_error"))
                continue

            qs = doc.get("quality_signals", {})
            if not isinstance(qs, dict):
                qs = {}

            reason = _check_quality(qs, **quality_kwargs)
            url = _extract_url(doc)
            results.append((group_id, url, reason))

    return results


# ---------------------------------------------------------------------------
# Phase 2: robots.txt check (single process)
# ---------------------------------------------------------------------------

def _check_robots_batch(
    urls_by_gid: Dict[int, str],
    robots_parquet: str,
    num_workers: int = 16,
) -> Dict[int, str]:
    """Check URLs against robots.txt using host-level precomputation.

    1. Precompute which hosts block root ``/`` → instant reject for all URLs on that host.
    2. Only fall back to per-URL checking for hosts with partial rules.
    """
    from robotrace import BLOCKED_BOTS, RobotsIndex
    from urllib.parse import urlsplit

    logger.info(f"Loading robots snapshot from {robots_parquet}")
    index = RobotsIndex.from_parquet(robots_parquet)
    logger.info(f"Loaded {index.num_hosts:,} hosts")

    # Precompute host-level decisions (one protego check per host, not per URL)
    logger.info("Precomputing host-level decisions...")
    blocked_hosts, partial_hosts = index.precompute_host_decisions(BLOCKED_BOTS)
    logger.info(
        f"Host decisions: {len(blocked_hosts):,} fully blocked, "
        f"{len(partial_hosts):,} partial rules, "
        f"{index.num_hosts - len(blocked_hosts) - len(partial_hosts):,} fully open"
    )

    # Fast check: host lookup for most URLs, per-URL only for partial hosts
    blocked: Dict[int, str] = {}
    n_fast = 0
    n_slow = 0
    for gid, url in urls_by_gid.items():
        if not url:
            continue
        try:
            host = urlsplit(url).hostname
            if not host:
                continue
            host = host.strip().lower().rstrip(".")

            if host in blocked_hosts:
                blocked[gid] = "robots"
                n_fast += 1
            elif host in partial_hosts:
                if not index.is_allowed_for_all(url, user_agents=BLOCKED_BOTS):
                    blocked[gid] = "robots"
                n_slow += 1
        except Exception:
            pass

    logger.info(
        f"Robots: {len(blocked):,} blocked "
        f"({n_fast:,} fast host-level, {n_slow:,} per-URL checks)"
    )
    return blocked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def clean_manifest(
    manifest_path: str,
    output_path: str,
    *,
    robots_parquet: Optional[str] = None,
    min_text_blocks: int = 1,
    min_avg_tokens_per_block: float = 20.0,
    min_total_tokens: int = 10,
    num_workers: int = 32,
) -> dict:
    """Filter a PIN-200M manifest by quality scores + robots.txt."""
    manifest = pq.read_table(manifest_path)
    n_total = len(manifest)

    jsonl_paths = manifest.column("jsonl_path").to_pylist()
    line_starts = manifest.column("line_start").to_numpy()
    line_lengths = manifest.column("line_length").to_numpy()
    group_ids = manifest.column("group_id").to_numpy()
    image_indices = manifest.column("image_index").to_numpy()

    first_rows = np.where(image_indices == 0)[0]
    n_docs = len(first_rows)
    logger.info(f"Manifest: {n_total:,} rows, {n_docs:,} documents")

    # Group by JSONL file
    file_docs: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for row_idx in first_rows:
        file_docs[jsonl_paths[row_idx]].append((
            int(line_starts[row_idx]),
            int(line_lengths[row_idx]),
            int(group_ids[row_idx]),
        ))
    logger.info(f"Grouped into {len(file_docs):,} JSONL files")

    quality_kwargs = {
        "min_text_blocks": min_text_blocks,
        "min_avg_tokens_per_block": min_avg_tokens_per_block,
        "min_total_tokens": min_total_tokens,
    }

    # Phase 1: parallel quality extraction + URL collection
    rejected: Dict[int, str] = {}
    urls_to_check: Dict[int, str] = {}

    logger.info(f"Phase 1: extracting quality signals ({num_workers} workers)")
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_process_file, path, docs, quality_kwargs): path
            for path, docs in file_docs.items()
        }
        done = 0
        for future in as_completed(futures):
            for gid, url, reason in future.result():
                if reason:
                    rejected[gid] = reason
                elif url:
                    urls_to_check[gid] = url
            done += 1
            if done % 100 == 0 or done == len(futures):
                logger.info(f"  Files: {done}/{len(futures)}, quality_rejected: {len(rejected):,}")

    logger.info(f"Phase 1 done: {len(rejected):,} rejected by quality, {len(urls_to_check):,} URLs to check")

    # Phase 2: robots.txt (single process, loaded once)
    if robots_parquet and urls_to_check:
        try:
            robots_rejected = _check_robots_batch(urls_to_check, robots_parquet, num_workers=num_workers)
            rejected.update(robots_rejected)
        except ImportError:
            logger.warning("robotrace not installed — skipping robots.txt filtering")

    # Filter manifest
    rejected_set = set(rejected.keys())
    keep_mask = np.array([int(gid) not in rejected_set for gid in group_ids])
    filtered = manifest.filter(keep_mask)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, output_path)

    # Save rejected doc IDs for rebuild-time filtering
    rejected_ids_path = str(Path(output_path).with_name("rejected_doc_ids.npy"))
    np.save(rejected_ids_path, np.array(sorted(rejected_set), dtype=np.int64))
    logger.info(f"Saved {len(rejected_set):,} rejected doc IDs to {rejected_ids_path}")

    # Stats
    reason_counts = defaultdict(int)
    for reason in rejected.values():
        reason_counts[reason] += 1

    n_kept_rows = len(filtered)
    stats = {
        "total_rows": n_total,
        "total_documents": n_docs,
        "rejected_documents": len(rejected),
        "kept_documents": n_docs - len(rejected),
        "kept_rows": n_kept_rows,
        "rejected_rows": n_total - n_kept_rows,
        "reasons": dict(reason_counts),
        "output": output_path,
    }

    logger.info(
        f"Result: {stats['kept_documents']:,}/{n_docs:,} documents kept "
        f"({n_kept_rows:,}/{n_total:,} rows)"
    )
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {reason}: {count:,}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="Input manifest parquet")
    parser.add_argument("--output", required=True, help="Output cleaned manifest parquet")
    parser.add_argument("--robots-parquet", default=None, help="Robots.txt snapshot parquet")
    parser.add_argument("--min-text-blocks", type=int, default=1)
    parser.add_argument("--min-avg-tokens-per-block", type=float, default=20.0)
    parser.add_argument("--min-total-tokens", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    stats = clean_manifest(
        manifest_path=args.manifest,
        output_path=args.output,
        robots_parquet=args.robots_parquet,
        min_text_blocks=args.min_text_blocks,
        min_avg_tokens_per_block=args.min_avg_tokens_per_block,
        min_total_tokens=args.min_total_tokens,
        num_workers=args.num_workers,
    )

    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
