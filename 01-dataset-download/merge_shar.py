#!/usr/bin/env python3
"""Merge existing Lhotse SHAR shards into fewer larger shards.

This script repacks SHAR *without* re-running decode/resample/tokenization.
It preserves field order and alignment from ``shar_index.json`` so
``CutSet.from_shar(...)`` remains valid.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger("merge_shar")
SHAR_NAME_RE = re.compile(r"^(?P<field>.+)\.(?P<idx>\d{6})(?P<suffix>\..+)$")


@dataclass(frozen=True)
class FieldMeta:
    suffix: str
    kind: str  # one of: tar, tar.gz, jsonl, jsonl.gz


@dataclass(frozen=True)
class FieldTask:
    dst_name: str
    src_paths: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class MergeTask:
    out_idx: int
    output_dir: str
    fields: tuple[FieldTask, ...]


def _parse_name(name: str) -> tuple[str, int, str]:
    match = SHAR_NAME_RE.match(name)
    if match is None:
        raise ValueError(
            f"Invalid SHAR shard filename: {name}. "
            "Expected '<field>.000000.<ext>'."
        )
    return match.group("field"), int(match.group("idx")), match.group("suffix")


def _kind_from_suffix(suffix: str) -> str:
    if suffix == ".tar":
        return "tar"
    if suffix == ".tar.gz":
        return "tar.gz"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".jsonl.gz":
        return "jsonl.gz"
    raise ValueError(
        f"Unsupported SHAR field suffix: {suffix}. "
        "Supported: .tar, .tar.gz, .jsonl, .jsonl.gz"
    )


def _load_index(shar_dir: Path, index_filename: str) -> dict[str, list[str]]:
    index_path = shar_dir / index_filename
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing SHAR index: {index_path}")

    payload = json.loads(index_path.read_text())
    fields = payload.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise ValueError(f"Invalid SHAR index (missing fields): {index_path}")

    normalized: dict[str, list[str]] = {}
    for field, rels in fields.items():
        if not isinstance(field, str) or not field:
            raise ValueError(f"Invalid field name in {index_path}: {field!r}")
        if not isinstance(rels, list) or not rels:
            raise ValueError(f"Field '{field}' has no shard entries in {index_path}")
        out: list[str] = []
        for rel in rels:
            rel_path = Path(rel)
            if rel_path.is_absolute():
                raise ValueError(f"Absolute path is not allowed in SHAR index: {rel}")
            parsed_field, _, _ = _parse_name(rel_path.name)
            if parsed_field != field:
                raise ValueError(
                    f"Field mismatch in SHAR index: entry '{rel}' belongs to '{parsed_field}', "
                    f"not '{field}'"
                )
            abs_path = (shar_dir / rel_path).resolve()
            if not abs_path.is_file():
                raise FileNotFoundError(f"Missing shard file referenced by index: {abs_path}")
            out.append(rel)
        normalized[field] = out
    return normalized


def _infer_field_meta(index: dict[str, list[str]]) -> dict[str, FieldMeta]:
    meta: dict[str, FieldMeta] = {}
    for field, rels in index.items():
        suffixes = {Path(rel).name and _parse_name(Path(rel).name)[2] for rel in rels}
        if len(suffixes) != 1:
            raise ValueError(
                f"Field '{field}' mixes file suffixes {sorted(suffixes)}; "
                "cannot merge safely."
            )
        suffix = next(iter(suffixes))
        meta[field] = FieldMeta(suffix=suffix, kind=_kind_from_suffix(suffix))
    return meta


def _validate_alignment(index: dict[str, list[str]]) -> int:
    fields = list(index.keys())
    num = len(index[fields[0]])
    for field in fields[1:]:
        if len(index[field]) != num:
            raise ValueError(
                f"Field '{field}' has {len(index[field])} shards, expected {num}. "
                "All fields must have aligned shard lists."
            )
    return num


def _group_by_fixed_count(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _copy_tarinfo(member: tarfile.TarInfo) -> tarfile.TarInfo:
    out = tarfile.TarInfo(name=member.name)
    out.size = member.size
    out.mode = member.mode
    out.uid = member.uid
    out.gid = member.gid
    out.mtime = member.mtime
    out.type = member.type
    out.linkname = member.linkname
    out.uname = member.uname
    out.gname = member.gname
    out.devmajor = member.devmajor
    out.devminor = member.devminor
    out.pax_headers = dict(member.pax_headers or {})
    return out


def _merge_tar_files(srcs: list[Path], dst: Path, gz: bool) -> None:
    mode = "w:gz" if gz else "w"
    with tarfile.open(dst, mode) as out_tf:
        for src in srcs:
            with tarfile.open(src, "r:*") as in_tf:
                for member in in_tf:
                    if not member.isfile():
                        continue
                    fd = in_tf.extractfile(member)
                    if fd is None:
                        continue
                    out_tf.addfile(_copy_tarinfo(member), fd)


def _concat_text(srcs: list[Path], dst: Path) -> None:
    with dst.open("wt", encoding="utf-8") as out_f:
        for src in srcs:
            with src.open("rt", encoding="utf-8", errors="replace") as in_f:
                shutil.copyfileobj(in_f, out_f, length=1024 * 1024)


def _concat_gzip_members(srcs: list[Path], dst: Path) -> None:
    # Concatenated gzip members are valid gzip streams and remain fully
    # readable by gzip libraries without recompressing.
    with dst.open("wb") as out_f:
        for src in srcs:
            with src.open("rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=1024 * 1024)


def _write_atomic(dst: Path, writer) -> None:
    tmp = Path(f"{dst}.tmp.{os.getpid()}")
    try:
        writer(tmp)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _write_field(srcs: list[Path], dst: Path, kind: str) -> None:
    if kind == "tar":
        _write_atomic(dst, lambda tmp: _merge_tar_files(srcs, tmp, gz=False))
        return
    if kind == "tar.gz":
        _write_atomic(dst, lambda tmp: _merge_tar_files(srcs, tmp, gz=True))
        return
    if kind == "jsonl":
        _write_atomic(dst, lambda tmp: _concat_text(srcs, tmp))
        return
    if kind == "jsonl.gz":
        _write_atomic(dst, lambda tmp: _concat_gzip_members(srcs, tmp))
        return
    raise ValueError(f"Unsupported field kind: {kind}")


def _merge_task(task: MergeTask) -> int:
    out_root = Path(task.output_dir)
    for field_task in task.fields:
        dst = out_root / field_task.dst_name
        srcs = [Path(p) for p in field_task.src_paths]
        _write_field(srcs, dst, field_task.kind)
    return task.out_idx


def _build_tasks(
    *,
    input_dir: Path,
    output_dir: Path,
    fields: list[str],
    field_meta: dict[str, FieldMeta],
    index: dict[str, list[str]],
    groups: list[list[int]],
) -> tuple[list[MergeTask], dict[str, list[str]]]:
    tasks: list[MergeTask] = []
    out_index = {field: [] for field in fields}

    for out_idx, group in enumerate(groups):
        field_tasks: list[FieldTask] = []
        for field in fields:
            meta = field_meta[field]
            dst_name = f"{field}.{out_idx:06d}{meta.suffix}"
            src_paths = tuple(str(input_dir / index[field][i]) for i in group)
            field_tasks.append(
                FieldTask(
                    dst_name=dst_name,
                    src_paths=src_paths,
                    kind=meta.kind,
                )
            )
            out_index[field].append(dst_name)
        tasks.append(
            MergeTask(
                out_idx=out_idx,
                output_dir=str(output_dir),
                fields=tuple(field_tasks),
            )
        )
    return tasks, out_index


def _copy_sidecars(input_dir: Path, output_dir: Path, sidecars: list[str]) -> None:
    """Copy selected top-level sidecar files/directories from input to output."""
    for name in sidecars:
        if "/" in name or name in (".", ".."):
            raise ValueError(
                f"Invalid sidecar name '{name}'. Use top-level names only "
                "(e.g. _manifests, _PREPARE_STATE.json)."
            )
        src = input_dir / name
        dst = output_dir / name
        if not src.exists():
            LOG.warning("Skipping missing sidecar: %s", src)
            continue
        if src.is_dir():
            shutil.copytree(src, dst)
            LOG.info("Copied sidecar dir: %s", name)
        else:
            shutil.copy2(src, dst)
            LOG.info("Copied sidecar file: %s", name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SHAR shards into fewer larger shards without reconversion."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--index-filename",
        type=str,
        default="shar_index.json",
        help="Input/output SHAR index filename (default: shar_index.json).",
    )
    parser.add_argument("--target-shards", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--copy-sidecars",
        nargs="*",
        default=[],
        help=(
            "Optional top-level files/dirs to copy from input to output after merge "
            "(e.g. _manifests _PREPARE_STATE.json)."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.target_shards <= 0:
        raise ValueError("--target-shards must be > 0")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be > 0")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)
    _validate_args(args)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("--output-dir must be different from --input-dir")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    index = _load_index(input_dir, args.index_filename)
    field_meta = _infer_field_meta(index)
    fields = list(index.keys())
    num_shards = _validate_alignment(index)
    shard_ids = list(range(num_shards))

    LOG.info("Input fields: %s", ", ".join(fields))
    LOG.info("Input shard count: %d", num_shards)

    shards_per_output = max(1, math.ceil(num_shards / args.target_shards))
    LOG.info(
        "Derived shards-per-output=%d from target-shards=%d",
        shards_per_output,
        args.target_shards,
    )
    groups = _group_by_fixed_count(shard_ids, shards_per_output)

    LOG.info("Output shard count: %d", len(groups))
    LOG.info("Reduction factor: %.2fx", num_shards / max(1, len(groups)))
    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, out_index = _build_tasks(
        input_dir=input_dir,
        output_dir=output_dir,
        fields=fields,
        field_meta=field_meta,
        index=index,
        groups=groups,
    )

    workers = min(args.num_workers, len(tasks))
    LOG.info("Merging with %d worker(s)", workers)

    if workers == 1:
        for done, task in enumerate(tasks, start=1):
            _merge_task(task)
            if done % 10 == 0 or done == len(tasks):
                LOG.info("Merged %d/%d output shards", done, len(tasks))
    else:
        done = 0
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=workers) as pool:
            for _ in pool.imap_unordered(_merge_task, tasks):
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    LOG.info("Merged %d/%d output shards", done, len(tasks))

    index_path = output_dir / args.index_filename
    index_path.write_text(json.dumps({"version": 1, "fields": out_index}, indent=2) + "\n")

    if args.copy_sidecars:
        _copy_sidecars(input_dir, output_dir, args.copy_sidecars)

    (output_dir / "_SUCCESS").write_text("ok\n")
    LOG.info("Wrote %s", index_path)
    LOG.info("Done: %s", output_dir)


if __name__ == "__main__":
    main()
