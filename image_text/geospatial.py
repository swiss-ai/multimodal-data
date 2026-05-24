#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import webdataset as wds
from PIL import Image

DEFAULT_OUTPUT_ROOT = Path("/tmp/shared")
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


def normalize_ext(ext: str) -> str:
    normalized = ext.lower().lstrip(".")
    if normalized == "jpeg":
        return "jpg"
    return normalized


def ext_from_suffix(suffix: str) -> str:
    return normalize_ext(suffix.split(".")[-1])


def ensure_generic_placeholders(caption: str, num_images: int) -> str:
    missing = [token for token in (f"<|img{i}|>" for i in range(1, num_images + 1)) if token not in caption]
    if not missing:
        return caption
    return "\n".join([*missing, caption])


def ensure_swisstopo_prefix(caption: str) -> str:
    # prefix = "<img1> <img2>"
    prefix = "Map: <img1>\nSatellite: <img2>"
    normalized = caption.replace("<map>", "<img1>").replace("<sat>", "<img2>")
    if normalized.startswith(prefix):
        return normalized
    return f"{prefix}\n{normalized}"


def load_captions(caption_dir: Path, pattern: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(caption_dir.glob(pattern)):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                sample_id = record["sample_id"]
                if sample_id in records:
                    raise ValueError(f"Duplicate sample_id {sample_id!r} in {path}:{line_no}")
                records[sample_id] = record
    return records


def iter_swisstopo_samples(dataset_dir: Path, captions: dict[str, dict]) -> Iterator[Sample]:
    tar_paths = [str(path) for path in sorted(dataset_dir.glob("*.tar"))]
    dataset = wds.WebDataset(tar_paths, shardshuffle=False, empty_check=False)
    for sample in dataset:
        key = sample["__key__"]
        record = captions.get(key)
        if record is None:
            continue
        sat = sample.get("sat.png")
        map_img = sample.get("map.png")
        if sat is None or map_img is None:
            source = sample.get("__url__", "<unknown>")
            raise ValueError(f"Missing map/sat pair for {key} in {source}")
        yield Sample(
            key=key,
            text=ensure_swisstopo_prefix(record["caption"]),
            images=(
                ImagePayload("img1.png", map_img),
                ImagePayload("img2.png", sat),
            ),
        )


def iter_ign_samples(dataset_dir: Path, captions: dict[str, dict]) -> Iterator[Sample]:
    tar_paths = [str(path) for path in sorted(dataset_dir.glob("*.tar"))]
    dataset = wds.WebDataset(tar_paths, shardshuffle=False, empty_check=False)
    for sample in dataset:
        key = sample["__key__"]
        record = captions.get(key)
        if record is None:
            continue
        image_suffix, image_bytes = next(
            (
                (suffix, data)
                for suffix, data in sample.items()
                if not suffix.startswith("__") and ext_from_suffix(suffix) in {"png", "jpg", "tif", "tiff"}
            ),
            (None, None),
        )
        if image_suffix is None or image_bytes is None:
            source = sample.get("__url__", "<unknown>")
            raise ValueError(f"Missing image payload for {key} in {source}")
        ext = ext_from_suffix(image_suffix)
        yield Sample(
            key=key,
            text=ensure_generic_placeholders(record["caption"], 1),
            images=(ImagePayload(f"img1.{ext}", image_bytes),),
        )


def convert_tiff_rgbi_to_rgb_tiff(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        arr = np.array(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels, got shape {arr.shape}")
    rgb = arr[:, :, :3]
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, format="TIFF")
    return out.getvalue()


def build_flair_member_lookup(captions: dict[str, dict]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for record in captions.values():
        source_path = record["source_path"]
        zip_name, _, member_name = source_path.partition(":")
        if not member_name:
            raise ValueError(f"Invalid source_path {source_path!r}")
        key = (Path(zip_name).name, member_name)
        lookup[key] = record
    return lookup


def iter_flair_samples(dataset_dir: Path, captions: dict[str, dict]) -> Iterator[Sample]:
    lookup = build_flair_member_lookup(captions)
    zip_paths = sorted(dataset_dir.glob("*.zip"))
    for zip_path in zip_paths:
        with ZipFile(zip_path) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith((".tif", ".tiff")):
                    continue
                record = lookup.get((zip_path.name, info.filename))
                if record is None:
                    continue
                rgb_tiff = convert_tiff_rgbi_to_rgb_tiff(zf.read(info))
                yield Sample(
                    key=record["sample_id"].replace("/", "__"),
                    text=ensure_generic_placeholders(record["caption"], 1),
                    images=(ImagePayload("img1.tif", rgb_tiff),),
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

    count = 0
    pattern = str(output_dir / "part-%06d.tar")
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


def validate_counts(expected: dict[str, dict], seen_keys: set[str], dataset_name: str) -> None:
    missing = set(expected) - seen_keys
    if missing:
        preview = ", ".join(sorted(list(missing)[:5]))
        raise ValueError(f"{dataset_name}: missing {len(missing)} captioned samples; first few: {preview}")


def build_swisstopo(output_root: Path, samples_per_shard: int, limit: int | None) -> int:
    captions = load_captions(
        Path("/tmp/shared/captions/Swisstopo"),
        "captions_*.jsonl",
    )
    seen: set[str] = set()

    def samples() -> Iterator[Sample]:
        for sample in iter_swisstopo_samples(
            Path("/path/to/data/vision-datasets/swisstopo/paired"),
            captions,
        ):
            seen.add(sample.key)
            yield sample

    count = write_dataset(
        samples(),
        output_root / "swisstopo_paired_webdataset",
        samples_per_shard,
        limit,
    )
    if limit is None:
        validate_counts(captions, seen, "Swisstopo")
    return count


def build_ign(output_root: Path, samples_per_shard: int, limit: int | None) -> int:
    captions = load_captions(
        Path("/tmp/shared/captions/IGN"),
        "caption_*.jsonl",
    )
    seen: set[str] = set()

    def samples() -> Iterator[Sample]:
        for sample in iter_ign_samples(
            Path("/path/to/data/vision-datasets/ign_city_tiles"),
            captions,
        ):
            seen.add(sample.key)
            yield sample

    count = write_dataset(
        samples(),
        output_root / "ign_city_tiles_webdataset",
        samples_per_shard,
        limit,
    )
    if limit is None:
        validate_counts(captions, seen, "IGN")
    return count


def build_flair(output_root: Path, samples_per_shard: int, limit: int | None) -> int:
    captions = load_captions(
        Path("/tmp/shared/captions/IGNF--FLAIR-HUB"),
        "caption_*.jsonl",
    )
    seen: set[str] = set()

    def samples() -> Iterator[Sample]:
        for sample in iter_flair_samples(
            Path(
                "/path/to/data/vision-datasets/hf_hub_cache/"
                "datasets--IGNF--FLAIR-HUB/snapshots/"
                "4cf55f57fd468fbd802681687c529a98c1274ce1/data"
            ),
            captions,
        ):
            seen.add(sample.key)
            yield sample

    count = write_dataset(
        samples(),
        output_root / "ignf_flair_hub_webdataset",
        samples_per_shard,
        limit,
    )
    if limit is None:
        validate_counts(
            {record["sample_id"].replace("/", "__"): record for record in captions.values()},
            seen,
            "IGNF/FLAIR-HUB",
        )
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build caption-paired webdatasets for Swisstopo, IGN, and IGNF/FLAIR-HUB."
    )
    parser.add_argument(
        "--dataset",
        choices=("swisstopo", "ign", "flair", "all"),
        default="all",
        help="Dataset to build.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for dataset-specific output directories.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=DEFAULT_SAMPLES_PER_SHARD,
        help="Maximum number of samples per output tar shard.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builders = {
        "swisstopo": build_swisstopo,
        "ign": build_ign,
        "flair": build_flair,
    }
    if args.dataset == "all":
        selected = list(builders.items())
    else:
        selected = [(args.dataset, builders[args.dataset])]

    for name, builder in selected:
        count = builder(args.output_root, args.samples_per_shard, args.limit)
        print(f"{name}: wrote {count} samples")


if __name__ == "__main__":
    main()
