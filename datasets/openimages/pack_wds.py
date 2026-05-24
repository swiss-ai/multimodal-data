#!/usr/bin/env python3
"""Pack OpenImages directories into flat WebDataset tar shards."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_SPLITS = ("train", "validation", "test")


@dataclass
class Sample:
    split: str
    image_path: Path

    @property
    def image_id(self) -> str:
        return self.image_path.stem

    @property
    def key(self) -> str:
        return f"{self.split}-{self.image_id}"


class ShardWriter:
    def __init__(self, output_root: Path, split: str, max_count: int) -> None:
        self.output_root = output_root
        self.split = split
        self.max_count = max_count
        self.shard_index = 0
        self.samples_in_shard = 0
        self.total_samples = 0
        self.current_path: Path | None = None
        self.current_tar: tarfile.TarFile | None = None

    def open_next_shard(self) -> None:
        if self.current_tar is not None:
            self.current_tar.close()

        self.current_path = self.output_root / f"{self.split}-{self.shard_index:06d}.tar"
        self.current_tar = tarfile.open(self.current_path, "w")
        self.shard_index += 1
        self.samples_in_shard = 0
        print(f"Opened {self.current_path}", flush=True)

    def ensure_open(self) -> None:
        if self.current_tar is None or self.samples_in_shard >= self.max_count:
            self.open_next_shard()

    def add_bytes(self, name: str, payload: bytes) -> None:
        assert self.current_tar is not None
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        self.current_tar.addfile(info, io.BytesIO(payload))

    def add_sample(self, sample: Sample) -> None:
        self.ensure_open()
        assert self.current_tar is not None

        jpg_name = f"{sample.key}.jpg"
        json_name = f"{sample.key}.json"

        file_size = sample.image_path.stat().st_size
        with sample.image_path.open("rb") as handle:
            info = tarfile.TarInfo(name=jpg_name)
            info.size = file_size
            self.current_tar.addfile(info, handle)

        metadata = {
            "__key__": sample.key,
            "split": sample.split,
            "image_id": sample.image_id,
            "source_path": f"{sample.split}/{sample.image_path.name}",
        }
        self.add_bytes(json_name, json.dumps(metadata, separators=(",", ":")).encode("utf-8"))

        self.samples_in_shard += 1
        self.total_samples += 1
        if self.total_samples % 10000 == 0:
            print(
                f"Packed split={sample.split} samples={self.total_samples} shards={self.shard_index}",
                flush=True,
            )

    def close(self) -> None:
        if self.current_tar is not None:
            self.current_tar.close()
            self.current_tar = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/tmp/data/openimages"),
        help="Directory containing train/, validation/, and test/ folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where flat tar shards will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to pack.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=10000,
        help="Maximum number of samples per tar shard.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on samples per split for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing tar shards for the selected splits before writing.",
    )
    return parser.parse_args()


def iter_samples(split_dir: Path, split: str, limit: int) -> Iterable[Sample]:
    count = 0
    for image_path in sorted(split_dir.glob("*.jpg")):
        yield Sample(split=split, image_path=image_path)
        count += 1
        if limit and count >= limit:
            return


def remove_existing_shards(output_root: Path, split: str) -> None:
    for path in sorted(output_root.glob(f"{split}-*.tar")):
        path.unlink()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict[str, int | list[str]]] = {"splits": {}}

    for split in args.splits:
        split_dir = args.input_root / split
        if not split_dir.is_dir():
            raise SystemExit(f"Split directory not found: {split_dir}")

        if args.overwrite:
            remove_existing_shards(args.output_root, split)

        writer = ShardWriter(output_root=args.output_root, split=split, max_count=args.max_count)
        try:
            for sample in iter_samples(split_dir, split, args.limit):
                writer.add_sample(sample)
        finally:
            writer.close()

        shard_paths = sorted(str(path.name) for path in args.output_root.glob(f"{split}-*.tar"))
        manifest["splits"][split] = {
            "samples": writer.total_samples,
            "shards": len(shard_paths),
            "files": shard_paths,
        }
        print(
            f"Finished split={split} samples={writer.total_samples} shards={len(shard_paths)}",
            flush=True,
        )

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
