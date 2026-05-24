#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import shutil
import sys
import tarfile
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

WHITESPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
BLOCK_SPLIT_RE = re.compile(r"\n\s*\n+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect suspicious repetition in webdataset .txt payloads. "
            "Findings are written to stdout as tab-separated lines: "
            "kind<TAB>tar_path<TAB>txt_member<TAB>details."
        )
    )
    parser.add_argument("datasets", nargs="+", help="Dataset directories containing part-*.tar shards")
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Directory for temporary duplicate-hash buckets. Defaults to the system temp directory.",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=24,
        help="Word n-gram size used for intra-caption repetition checks. Default: 24",
    )
    parser.add_argument(
        "--min-ngram-count",
        type=int,
        default=3,
        help="Minimum count for a repeated n-gram to be considered suspicious. Default: 3",
    )
    parser.add_argument(
        "--min-sentence-chars",
        type=int,
        default=60,
        help="Ignore repeated sentences shorter than this many characters after normalization. Default: 60",
    )
    parser.add_argument(
        "--min-sentence-repeat-count",
        type=int,
        default=3,
        help="Flag repeated sentences when any normalized sentence appears at least this many times. Default: 3",
    )
    parser.add_argument(
        "--min-sentence-excess-chars",
        type=int,
        default=200,
        help=(
            "Flag repeated sentences when the repeated excess text across all repeated sentences "
            "reaches at least this many characters. Default: 200"
        ),
    )
    parser.add_argument(
        "--min-unique-repeated-sentences",
        type=int,
        default=2,
        help="Flag repeated sentences when at least this many distinct sentences repeat. Default: 2",
    )
    parser.add_argument(
        "--min-ngram-repeat-coverage",
        type=int,
        default=80,
        help=("Flag repeated n-grams only when repeated spans cover at least this many words in total. Default: 80"),
    )
    parser.add_argument(
        "--min-ngram-max-count",
        type=int,
        default=3,
        help="Flag repeated n-grams only when the most repeated n-gram appears at least this many times. Default: 3",
    )
    parser.add_argument(
        "--limit-tars",
        type=int,
        default=None,
        help="Only inspect the first N tar shards per dataset. Useful for smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 288),
        help="Number of worker processes used to scan tar shards. Default: min(cpu_count, 288)",
    )
    return parser.parse_args()


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def iter_tar_paths(dataset_dir: Path, limit: int | None) -> list[Path]:
    tar_paths = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_file()
        and (path.name.endswith(".tar") or path.name.endswith(".tar.gz") or path.name.endswith(".tgz"))
    )
    if limit is not None:
        tar_paths = tar_paths[:limit]
    return tar_paths


def iter_txt_members(tar_path: Path):
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            yield member.name, extracted.read().decode("utf-8", errors="replace")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\x00", " ")
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized.casefold()


def split_text_blocks(text: str) -> list[tuple[str, str]]:
    stripped = text.strip()
    if not stripped:
        return [("full", "")]

    blocks = [block.strip() for block in BLOCK_SPLIT_RE.split(stripped) if block.strip()]
    parsed_blocks: list[tuple[str, str]] = []

    for block in blocks:
        lines = block.splitlines()
        if not lines or not lines[0].startswith("lang="):
            return [("full", stripped)]
        lang = lines[0][5:].strip() or "unknown"
        body = "\n".join(lines[1:]).strip()
        parsed_blocks.append((lang, body))

    return parsed_blocks or [("full", stripped)]


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for chunk in SENTENCE_SPLIT_RE.split(text):
        sentence = chunk.strip()
        if sentence:
            parts.append(sentence)
    return parts


def sentence_repetition_stats(text: str, min_sentence_chars: int) -> dict[str, int] | None:
    counts: Counter[str] = Counter()
    lengths: dict[str, int] = {}

    for sentence in split_sentences(text):
        normalized = normalize_text(sentence)
        if len(normalized) < min_sentence_chars:
            continue
        counts[normalized] += 1
        lengths.setdefault(normalized, len(normalized))

    repeated = {sentence: count for sentence, count in counts.items() if count >= 2}
    if not repeated:
        return None

    excess_chars = sum(lengths[sentence] * (count - 1) for sentence, count in repeated.items())
    return {
        "unique_repeated_sentences": len(repeated),
        "max_count": max(repeated.values()),
        "excess_chars": excess_chars,
    }


def ngram_repetition_stats(text: str, ngram_size: int, min_ngram_count: int) -> dict[str, int] | None:
    words = WORD_RE.findall(normalize_text(text))
    if len(words) < ngram_size * min_ngram_count:
        return None

    positions_by_ngram: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in range(len(words) - ngram_size + 1):
        positions_by_ngram[tuple(words[index : index + ngram_size])].append(index)

    repeated_positions = {
        ngram: positions for ngram, positions in positions_by_ngram.items() if len(positions) >= min_ngram_count
    }
    if not repeated_positions:
        return None

    intervals: list[tuple[int, int]] = []
    max_count = 0
    for positions in repeated_positions.values():
        max_count = max(max_count, len(positions))
        for start in positions:
            intervals.append((start, start + ngram_size))

    intervals.sort()
    merged_intervals: list[list[int]] = []
    for start, end in intervals:
        if not merged_intervals or start > merged_intervals[-1][1]:
            merged_intervals.append([start, end])
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], end)

    coverage_words = sum(end - start for start, end in merged_intervals)
    return {
        "unique_repeated_ngrams": len(repeated_positions),
        "max_count": max_count,
        "coverage_words": coverage_words,
    }


def inspect_internal_repetition(
    text: str,
    min_sentence_chars: int,
    ngram_size: int,
    min_ngram_count: int,
    min_sentence_repeat_count: int,
    min_sentence_excess_chars: int,
    min_unique_repeated_sentences: int,
    min_ngram_repeat_coverage: int,
    min_ngram_max_count: int,
) -> list[tuple[str, str]]:
    findings: list[tuple[str, str]] = []

    for block_name, block_text in split_text_blocks(text):
        sentence_stats = sentence_repetition_stats(block_text, min_sentence_chars=min_sentence_chars)
        if sentence_stats and (
            sentence_stats["max_count"] >= min_sentence_repeat_count
            or sentence_stats["excess_chars"] >= min_sentence_excess_chars
            or sentence_stats["unique_repeated_sentences"] >= min_unique_repeated_sentences
        ):
            findings.append(
                (
                    "repeated_sentence",
                    "block="
                    f"{block_name}\tunique_repeated_sentences={sentence_stats['unique_repeated_sentences']}"
                    f"\tmax_count={sentence_stats['max_count']}"
                    f"\texcess_chars={sentence_stats['excess_chars']}",
                )
            )

        ngram_stats = ngram_repetition_stats(
            block_text,
            ngram_size=ngram_size,
            min_ngram_count=min_ngram_count,
        )
        if ngram_stats and (
            ngram_stats["max_count"] >= min_ngram_max_count
            and ngram_stats["coverage_words"] >= min_ngram_repeat_coverage
        ):
            findings.append(
                (
                    "repeated_ngram",
                    "block="
                    f"{block_name}\tngram_size={ngram_size}"
                    f"\tunique_repeated_ngrams={ngram_stats['unique_repeated_ngrams']}"
                    f"\tmax_count={ngram_stats['max_count']}"
                    f"\tcoverage_words={ngram_stats['coverage_words']}",
                )
            )

    return findings


def make_bucket_writer(bucket_root: Path):
    handles: dict[str, object] = {}

    def write_entry(digest: str, tar_path: Path, member_name: str) -> None:
        bucket_name = f"{digest[:2]}.tsv"
        handle = handles.get(bucket_name)
        if handle is None:
            handle = open(bucket_root / bucket_name, "a", encoding="utf-8")
            handles[bucket_name] = handle
        handle.write(f"{digest}\t{tar_path}\t{member_name}\n")

    def close_all() -> None:
        for handle in handles.values():
            handle.close()

    return write_entry, close_all


def emit_duplicate_findings(bucket_root: Path) -> int:
    duplicate_lines = 0

    for bucket_path in sorted(bucket_root.glob("*.tsv")):
        groups: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)

        with open(bucket_path, "r", encoding="utf-8") as handle:
            for line in handle:
                digest, tar_path, member_name = line.rstrip("\n").split("\t", 2)
                groups[digest].append((tar_path, member_name))

        for entries in groups.values():
            if len(entries) < 2:
                continue
            group_size = len(entries)
            for tar_path, member_name in entries:
                print(f"duplicate_text\t{tar_path}\t{member_name}\tgroup_size={group_size}")
                duplicate_lines += 1

    return duplicate_lines


def scan_tar(
    tar_path: str,
    working_root: str,
    min_sentence_chars: int,
    ngram_size: int,
    min_ngram_count: int,
    min_sentence_repeat_count: int,
    min_sentence_excess_chars: int,
    min_unique_repeated_sentences: int,
    min_ngram_repeat_coverage: int,
    min_ngram_max_count: int,
) -> dict[str, object]:
    tar_path_obj = Path(tar_path)
    worker_id = hashlib.sha1(tar_path.encode("utf-8")).hexdigest()
    hash_path = Path(working_root) / "hashes" / f"{worker_id}.tsv"
    finding_path = Path(working_root) / "findings" / f"{worker_id}.tsv"

    txt_members = 0
    finding_lines = 0

    with (
        open(hash_path, "w", encoding="utf-8") as hash_handle,
        open(finding_path, "w", encoding="utf-8") as finding_handle,
    ):
        for member_name, text in iter_txt_members(tar_path_obj):
            txt_members += 1

            for kind, details in inspect_internal_repetition(
                text,
                min_sentence_chars=min_sentence_chars,
                ngram_size=ngram_size,
                min_ngram_count=min_ngram_count,
                min_sentence_repeat_count=min_sentence_repeat_count,
                min_sentence_excess_chars=min_sentence_excess_chars,
                min_unique_repeated_sentences=min_unique_repeated_sentences,
                min_ngram_repeat_coverage=min_ngram_repeat_coverage,
                min_ngram_max_count=min_ngram_max_count,
            ):
                finding_handle.write(f"{kind}\t{tar_path_obj}\t{member_name}\t{details}\n")
                finding_lines += 1

            digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
            hash_handle.write(f"{digest}\t{tar_path_obj}\t{member_name}\n")

    return {
        "tar_path": tar_path,
        "txt_members": txt_members,
        "finding_lines": finding_lines,
        "hash_path": str(hash_path),
        "finding_path": str(finding_path),
    }


def flush_worker_findings(finding_path: Path) -> None:
    if not finding_path.exists():
        return
    with open(finding_path, "r", encoding="utf-8") as handle:
        for line in handle:
            sys.stdout.write(line)
    finding_path.unlink(missing_ok=True)


def bucketize_hash_files(hash_paths: list[Path], bucket_root: Path) -> None:
    write_bucket_entry, close_bucket_files = make_bucket_writer(bucket_root)
    try:
        for hash_path in hash_paths:
            with open(hash_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    digest, tar_path, member_name = line.rstrip("\n").split("\t", 2)
                    write_bucket_entry(digest, Path(tar_path), member_name)
    finally:
        close_bucket_files()


def inspect_dataset(
    dataset_dir: Path,
    tmp_dir: str | None,
    min_sentence_chars: int,
    ngram_size: int,
    min_ngram_count: int,
    limit_tars: int | None,
    workers: int,
    min_sentence_repeat_count: int,
    min_sentence_excess_chars: int,
    min_unique_repeated_sentences: int,
    min_ngram_repeat_coverage: int,
    min_ngram_max_count: int,
) -> int:
    tar_paths = iter_tar_paths(dataset_dir, limit=limit_tars)
    if not tar_paths:
        eprint(f"[skip] no tar shards found in {dataset_dir}")
        return 0

    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f"detect_repetition_{dataset_dir.name}_",
            dir=tmp_dir,
        )
    )
    hash_root = temp_root / "hashes"
    finding_root = temp_root / "findings"
    bucket_root = temp_root / "buckets"
    hash_root.mkdir()
    finding_root.mkdir()
    bucket_root.mkdir()

    effective_workers = max(1, min(workers, len(tar_paths)))

    eprint(f"[scan] {dataset_dir} ({len(tar_paths)} tar shards, workers={effective_workers})")

    txt_members = 0
    finding_lines = 0
    hash_paths: list[Path] = []
    completed = 0

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    scan_tar,
                    str(tar_path),
                    str(temp_root),
                    min_sentence_chars,
                    ngram_size,
                    min_ngram_count,
                    min_sentence_repeat_count,
                    min_sentence_excess_chars,
                    min_unique_repeated_sentences,
                    min_ngram_repeat_coverage,
                    min_ngram_max_count,
                )
                for tar_path in tar_paths
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += 1
                txt_members += int(result["txt_members"])
                finding_lines += int(result["finding_lines"])
                hash_paths.append(Path(str(result["hash_path"])))
                flush_worker_findings(Path(str(result["finding_path"])))

                if completed == 1 or completed % 25 == 0 or completed == len(tar_paths):
                    eprint(
                        f"[scan] {dataset_dir.name}: {completed}/{len(tar_paths)} tar shards, {txt_members} txt files"
                    )

        bucketize_hash_files(hash_paths, bucket_root)
        duplicate_lines = emit_duplicate_findings(bucket_root)
        finding_lines += duplicate_lines
        eprint(
            f"[done] {dataset_dir.name}: {txt_members} txt files inspected, "
            f"{duplicate_lines} duplicate-text lines, {finding_lines} total findings"
        )
        return finding_lines
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    args = parse_args()
    total_findings = 0

    for dataset in args.datasets:
        total_findings += inspect_dataset(
            dataset_dir=Path(dataset).resolve(),
            tmp_dir=args.tmp_dir,
            min_sentence_chars=args.min_sentence_chars,
            ngram_size=args.ngram_size,
            min_ngram_count=args.min_ngram_count,
            limit_tars=args.limit_tars,
            workers=args.workers,
            min_sentence_repeat_count=args.min_sentence_repeat_count,
            min_sentence_excess_chars=args.min_sentence_excess_chars,
            min_unique_repeated_sentences=args.min_unique_repeated_sentences,
            min_ngram_repeat_coverage=args.min_ngram_repeat_coverage,
            min_ngram_max_count=args.min_ngram_max_count,
        )

    eprint(f"[summary] total findings: {total_findings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
