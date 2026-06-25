#!/usr/bin/env python3
"""Auto-resolve a JSONL's image refs to tar member names for jsonl+tar datasets.

When a JSONL+tar dataset's image field embeds an extra path prefix (e.g.
``"MSCOCO/images/foo.jpg"`` while the tar member is ``"images/foo.jpg"``, or
even an absolute path like ``"/data/tir/.../laion-multi-1M/images/foo.jpg"``),
a single global prefix-strip can't recover the correct tar member when different
subsets carry different prefixes. This tool rewrites the JSONL so the image
field points at real tar member names.

Strategy: **suffix matching per ref**. For each image ref:

  1. Strip leading "/" if absolute.
  2. Try the full ref as a tar member name; if not present, drop the first
     path component and try again; continue until match or exhausted.
  3. First match wins; the ref is rewritten to that member name.

Handles, uniformly:
  - relative refs with extra prefix (``MSCOCO/images/foo.jpg`` -> ``images/foo.jpg``)
  - absolute refs (``/data/.../images/foo.jpg`` -> ``images/foo.jpg``)
  - val-set leakage (refs that don't resolve in train tars -> drop)

Example (PangeaInstruct):

    python normalize_jsonl_tar.py \
        --jsonl-in   /path/PangeaIns_train.jsonl \
        --tar-root   /path/PangeaInstruct \
        --image-field image \
        --jsonl-out  /path/PangeaIns_train_normalized.jsonl \
        --strip-map-out /path/subset_strip_map.json \
        --drop-log   /path/dropped.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# tar member discovery (lightweight — names only, no image header reads)
# ---------------------------------------------------------------------------

def _members_of_one_tar(path: str) -> tuple[str, set[str]]:
    """Walk tar headers and return all file-member names (no payload reads)."""
    names: set[str] = set()
    try:
        with tarfile.open(path, "r") as tf:
            for m in tf:
                if m.isfile():
                    names.add(m.name)
    except Exception:
        logger.warning("skip unreadable tar: %s", path, exc_info=True)
    return path, names


def build_tar_member_set(
    tar_paths: Iterable[str],
    workers: int = 16,
) -> frozenset[str]:
    """Build the set of all file-member names across given tar archives.

    Members are deduplicated across archives — useful since refs are looked up
    by member name regardless of which tar they live in.
    """
    paths = list(tar_paths)
    logger.info("Indexing %d tar archives with %d workers", len(paths), workers)
    unified: set[str] = set()
    if not paths:
        return frozenset()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, names in pool.map(_members_of_one_tar, paths):
            unified.update(names)
            logger.info("  %s — %d members", path, len(names))
    return frozenset(unified)


def discover_tars(tar_root: str, glob_pattern: str = "**/*.tar*") -> list[str]:
    """Glob ``tar_root`` for tar archives (.tar, .tar.gz)."""
    pattern = os.path.join(tar_root, glob_pattern)
    paths = sorted(p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p))
    return paths


# ---------------------------------------------------------------------------
# per-subset strip rule detection
# ---------------------------------------------------------------------------

def _try_strip(ref: str, n: int) -> str | None:
    """Strip the first ``n`` path components from ``ref``."""
    parts = ref.split("/")
    if n >= len(parts):
        return None
    return "/".join(parts[n:])


def _normalize_refs_list(value) -> list[str]:
    """Coerce image field value to a list of relative-path strings.

    Accepts:
      - str: single image ref
      - list[str]: multi-image refs
      - anything else: returns []  (caller can decide to drop or keep)
    Drops absolute paths.
    """
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list) and all(isinstance(v, str) for v in value):
        items = value
    else:
        return []
    return [v for v in items if not v.startswith("/")]


def _sample_refs_by_subset(
    jsonl_path: str,
    image_field: str,
    samples_per_subset: int,
) -> dict[str, list[str]]:
    """Stream jsonl once, collect up to N refs per subset (first path component).

    Handles both single-image (str) and multi-image (list[str]) image fields.
    """
    samples: dict[str, list[str]] = defaultdict(list)
    with open(jsonl_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for ref in _normalize_refs_list(rec.get(image_field)):
                subset = ref.split("/", 1)[0] if "/" in ref else ref
                if len(samples[subset]) < samples_per_subset:
                    samples[subset].append(ref)
    return samples


def detect_subset_strip_rules(
    jsonl_path: str,
    tar_members: frozenset[str],
    image_field: str = "image",
    samples_per_subset: int = 20,
    max_strip: int = 4,
) -> tuple[dict[str, int], list[str]]:
    """Discover per-subset strip counts that make image refs resolve to tar members.

    Returns ``(rules, unresolved_subsets)`` where rules maps subset->strip_count.
    """
    samples = _sample_refs_by_subset(jsonl_path, image_field, samples_per_subset)
    logger.info(
        "Sampled %d records across %d subsets for strip-rule detection",
        sum(len(v) for v in samples.values()),
        len(samples),
    )

    rules: dict[str, int] = {}
    unresolved: list[str] = []
    for subset, refs in sorted(samples.items()):
        for n in range(max_strip + 1):
            stripped = [_try_strip(r, n) for r in refs]
            if any(s is None for s in stripped):
                continue
            if all(s in tar_members for s in stripped):
                rules[subset] = n
                logger.info("  subset %-35s strip=%d  (n_refs=%d)", subset, n, len(refs))
                break
        else:
            unresolved.append(subset)
            logger.warning(
                "  subset %-35s NO MATCH within strip 0..%d  (samples: %s)",
                subset, max_strip, refs[:3],
            )
    return rules, unresolved


# ---------------------------------------------------------------------------
# jsonl rewrite
# ---------------------------------------------------------------------------

def rewrite_image_refs(
    jsonl_in: str,
    jsonl_out: str,
    rules: dict[str, int],
    tar_members: frozenset[str],
    image_field: str = "image",
    drop_log: str | None = None,
) -> dict[str, int]:
    """Stream jsonl: rewrite image field per ``rules``, drop unresolvable rows.

    Rows with no ``image_field`` are kept as-is (text-only rows route through
    the scanner's text path).
    """
    kept = kept_text_only = 0
    drop_no_member = 0

    def _rewrite_one(ref: str) -> tuple[str | None, str]:
        """Per-ref suffix matching: strip leading components until a tar member matches.

        Strips a leading "/" first (absolute -> relative), then tries the full ref
        and progressively-suffix-stripped variants. First match wins. The per-subset
        ``rules`` dict (passed in for compatibility) is consulted only as a hint —
        if a subset has a known strip count, we try that first to skip iteration.
        """
        if ref.startswith("/"):
            ref = ref[1:]
        # Hint from auto-detected rules
        subset = ref.split("/", 1)[0] if "/" in ref else ref
        hint = rules.get(subset)
        n_components = ref.count("/") + 1
        # Build try-order: hint first (if any), then 0..max_strip_we_can_do
        try_order = []
        if hint is not None:
            try_order.append(hint)
        for n in range(min(n_components, 8)):
            if n != hint:
                try_order.append(n)
        for n in try_order:
            normalized = _try_strip(ref, n)
            if normalized is not None and normalized in tar_members:
                return normalized, ""
        return None, "no_tar_member"

    drops = open(drop_log, "w") if drop_log else None
    try:
        with open(jsonl_in) as src, open(jsonl_out, "w") as dst:
            for line in src:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    if drops:
                        drops.write(json.dumps({"reason": "bad_json", "raw": line[:200]}) + "\n")
                    continue

                value = rec.get(image_field)
                if value is None:
                    # text-only row — keep
                    dst.write(line)
                    kept_text_only += 1
                    continue

                # Coerce to list for uniform handling
                if isinstance(value, str):
                    items = [value]
                    is_list = False
                elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                    items = value
                    is_list = True
                else:
                    # weird shape — drop
                    if drops:
                        drops.write(json.dumps({"reason": "bad_image_field_type", "id": rec.get("id"), "image": value}) + "\n")
                    continue

                normalized_items: list[str] = []
                drop_reason: str | None = None
                for r in items:
                    norm, reason = _rewrite_one(r)
                    if norm is None:
                        drop_reason = reason
                        break
                    normalized_items.append(norm)

                if drop_reason:
                    # Drop the whole record if ANY ref can't be resolved (for
                    # multi-image SFT, conv expects all images to be loadable).
                    drop_no_member += 1
                    if drops:
                        drops.write(json.dumps({
                            "reason": drop_reason,
                            "id": rec.get("id"),
                            "image": value,
                        }) + "\n")
                    continue

                rec[image_field] = normalized_items if is_list else normalized_items[0]
                dst.write(json.dumps(rec, ensure_ascii=False))
                dst.write("\n")
                kept += 1
    finally:
        if drops:
            drops.close()

    return {
        "kept_image": kept,
        "kept_text_only": kept_text_only,
        "drop_no_tar_member": drop_no_member,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl-in", required=True)
    p.add_argument("--tar-root", required=True,
                   help="Directory to recursively scan for *.tar / *.tar.gz")
    p.add_argument("--tar-glob", default="**/*.tar*",
                   help="Glob under --tar-root (default: **/*.tar*)")
    p.add_argument("--image-field", default="image")
    p.add_argument("--jsonl-out", required=True)
    p.add_argument("--strip-map-out", default=None,
                   help="Optional JSON file to write the discovered subset->strip_count map")
    p.add_argument("--drop-log", default=None,
                   help="Optional JSONL file to log each dropped record with a reason")
    p.add_argument("--samples-per-subset", type=int, default=20,
                   help="Sample size per subset for strip-rule detection (default: 20)")
    p.add_argument("--max-strip", type=int, default=4,
                   help="Maximum leading components to try stripping (default: 4)")
    p.add_argument("--tar-index-workers", type=int, default=16,
                   help="Parallel workers for tar member indexing (default: 16)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("=== step 1/3: discover + index tars ===")
    paths = discover_tars(args.tar_root, args.tar_glob)
    if not paths:
        logger.error("No tar archives found under %s matching %s", args.tar_root, args.tar_glob)
        sys.exit(1)
    members = build_tar_member_set(paths, workers=args.tar_index_workers)
    logger.info("tar_member_set: %d unique member names across %d archives", len(members), len(paths))

    logger.info("=== step 2/3: detect per-subset strip rules ===")
    rules, unresolved = detect_subset_strip_rules(
        args.jsonl_in, members,
        image_field=args.image_field,
        samples_per_subset=args.samples_per_subset,
        max_strip=args.max_strip,
    )
    logger.info("Discovered %d subset rules, %d unresolved", len(rules), len(unresolved))
    if unresolved:
        logger.warning("unresolved subsets (rows will drop): %s", unresolved)
    if args.strip_map_out:
        Path(args.strip_map_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.strip_map_out, "w") as f:
            json.dump({"rules": rules, "unresolved": unresolved}, f, indent=2)
        logger.info("Wrote strip-rule map to %s", args.strip_map_out)

    logger.info("=== step 3/3: rewrite jsonl ===")
    Path(args.jsonl_out).parent.mkdir(parents=True, exist_ok=True)
    if args.drop_log:
        Path(args.drop_log).parent.mkdir(parents=True, exist_ok=True)
    stats = rewrite_image_refs(
        args.jsonl_in, args.jsonl_out, rules, members,
        image_field=args.image_field, drop_log=args.drop_log,
    )
    logger.info("Done. Stats:")
    for k, v in stats.items():
        logger.info("  %-25s %d", k, v)


if __name__ == "__main__":
    main()
