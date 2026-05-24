#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import webdataset as wds

DEFAULT_DATASET_DIR = Path(os.getenv("EGOPAT_DATASET_DIR", "/path/to/data"))
DEFAULT_CAPTION_DIR = Path(os.getenv("EGOPAT_CAPTION_DIR", "/path/to/captions"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("EGOPAT_OUTPUT_DIR", "/path/to/output"))
DEFAULT_SAMPLES_PER_SHARD = 10_000


@dataclass(frozen=True)
class ImagePayload:
    suffix: str
    data: bytes


@dataclass(frozen=True)
class Sample:
    key: str
    text: str
    images: tuple[ImagePayload, ...]


def normalize_placeholders(caption: str, num_images: int) -> str:
    for src_idx in range(num_images):
        caption = caption.replace(f"<|img{src_idx}|>", f"<|img{src_idx + 1}|>")

    missing = [token for token in (f"<|img{i}|>" for i in range(1, num_images + 1)) if token not in caption]
    if missing:
        return "\n".join([*missing, caption])
    return caption


def load_captions(caption_dir: Path) -> dict[str, str]:
    captions: dict[str, str] = {}
    for path in sorted(caption_dir.glob("captions_task*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict in {path}, got {type(payload).__name__}")
        for key, value in payload.items():
            if key in captions:
                raise ValueError(f"Duplicate caption key {key!r} in {path}")
            if not isinstance(value, str):
                raise ValueError(f"Expected string caption for {key!r} in {path}")
            captions[key] = value.strip()
    return captions


def iter_samples(dataset_dir: Path, captions: dict[str, str]) -> Iterator[Sample]:
    urls = [str(path) for path in sorted(dataset_dir.glob("shard-*.tar"))]
    if not urls:
        raise FileNotFoundError(f"No shard-*.tar files found in {dataset_dir}")

    dataset = wds.WebDataset(urls, shardshuffle=False, empty_check=False)
    for sample in dataset:
        key = sample["__key__"]
        caption = captions.get(key)
        if caption is None:
            continue

        image_keys = ("img0.jpg", "img1.jpg", "img2.jpg")
        image_payloads: list[ImagePayload] = []
        for output_idx, image_key in enumerate(image_keys, start=1):
            data = sample.get(image_key)
            if data is None:
                source = sample.get("__url__", "<unknown>")
                raise ValueError(f"Missing {image_key} for {key} in {source}")
            image_payloads.append(ImagePayload(f"img{output_idx}.jpg", data))

        yield Sample(
            key=key,
            text=normalize_placeholders(caption, 3),
            images=tuple(image_payloads),
        )


def write_dataset(
    samples: Iterable[Sample],
    output_dir: Path,
    samples_per_shard: int,
    limit: int | None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("part-*.tar"))
    if existing:
        raise FileExistsError(f"Output directory {output_dir} already contains shard files")

    pattern = str(output_dir / "part-%06d.tar")
    count = 0
    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as writer:
        for sample in samples:
            payload: dict[str, bytes | str] = {
                "__key__": sample.key,
                "txt": sample.text,
            }
            for image in sample.images:
                payload[image.suffix] = image.data
            writer.write(payload)
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def validate_counts(expected: dict[str, str], seen_keys: set[str]) -> None:
    missing = set(expected) - seen_keys
    if missing:
        preview = ", ".join(sorted(list(missing)[:5]))
        raise ValueError(f"Missing {len(missing)} captioned samples; first few: {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EgoPAT3Dv2 image-caption webdataset shards.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--caption-dir", type=Path, default=DEFAULT_CAPTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples-per-shard", type=int, default=DEFAULT_SAMPLES_PER_SHARD)
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captions = load_captions(args.caption_dir)
    seen: set[str] = set()

    def sample_iter() -> Iterator[Sample]:
        for sample in iter_samples(args.dataset_dir, captions):
            seen.add(sample.key)
            yield sample

    count = write_dataset(sample_iter(), args.output_dir, args.samples_per_shard, args.limit)
    if args.limit is None:
        validate_counts(captions, seen)
    print(f"wrote {count} samples")


if __name__ == "__main__":
    main()
