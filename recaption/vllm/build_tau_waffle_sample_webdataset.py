#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/tau-vailab___WAFFLE/parquet/permissive")
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "tau_waffle_architecture_gemma4_sample16"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "sample_webdataset"
DEFAULT_MANIFEST_PATH = OUTPUT_ROOT / "sample_manifest.jsonl"
DEFAULT_LIMIT = 16
DEFAULT_SEED = 0
SHARD_MAXCOUNT = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def list_parquet_paths() -> list[Path]:
    parquet_paths = sorted(DATASET_ROOT.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {DATASET_ROOT}")
    return parquet_paths


def decode_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_metadata(row: dict, parquet_path: Path, row_index: int) -> dict:
    metadata = {}
    for key, value in row.items():
        if key == "image_bytes":
            continue
        if isinstance(value, list):
            metadata[key] = [decode_text(item) for item in value if decode_text(item)]
            continue
        metadata[key] = decode_text(value)

    metadata["parquet_path"] = parquet_path.relative_to(DATASET_ROOT).as_posix()
    metadata["row_index"] = row_index
    return metadata


def detect_image_ext(image_bytes: bytes, image_path: str | None) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "webp"
    if image_path:
        suffix = Path(image_path).suffix.lower().lstrip(".")
        if suffix in {"jpg", "jpeg", "png", "webp"}:
            return "jpg" if suffix == "jpeg" else suffix
    raise ValueError("Unable to determine image extension")


def make_sample_id(parquet_path: Path, row_index: int) -> str:
    raw = f"{parquet_path.relative_to(DATASET_ROOT).as_posix()}:{row_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def choose_row_indices(num_rows: int, picks: int) -> list[int]:
    if picks <= 0:
        return []
    if picks >= num_rows:
        return list(range(num_rows))
    return [((2 * i + 1) * num_rows) // (2 * picks) for i in range(picks)]


def assign_counts(total_items: int, bucket_count: int) -> list[int]:
    base = total_items // bucket_count
    remainder = total_items % bucket_count
    return [base + (1 if i < remainder else 0) for i in range(bucket_count)]


def rotate_parquet_paths(parquet_paths: list[Path], seed: int) -> list[Path]:
    if not parquet_paths:
        return parquet_paths
    offset = seed % len(parquet_paths)
    return parquet_paths[offset:] + parquet_paths[:offset]


def build_sample_rows(limit: int, seed: int) -> list[dict]:
    parquet_paths = list_parquet_paths()
    file_count = min(limit, len(parquet_paths))
    selected_parquets = rotate_parquet_paths(parquet_paths, seed)[:file_count]
    counts = assign_counts(limit, file_count)
    sample_rows: list[dict] = []

    for parquet_path, count in zip(selected_parquets, counts, strict=True):
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        row_indices = choose_row_indices(total_rows, count)
        table = parquet_file.read(columns=None)
        rows = table.to_pylist()
        for row_index in row_indices:
            row = rows[row_index]
            image_bytes = row.get("image_bytes")
            if not isinstance(image_bytes, (bytes, bytearray)):
                raise ValueError(f"Missing image_bytes for {parquet_path.relative_to(DATASET_ROOT)}:{row_index}")

            metadata = normalize_metadata(row, parquet_path, row_index)
            sample_id = make_sample_id(parquet_path, row_index)
            image_ext = detect_image_ext(bytes(image_bytes), metadata.get("image_path"))
            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "image_ext": image_ext,
                    "image_bytes": bytes(image_bytes),
                    "metadata": metadata,
                }
            )

    if len(sample_rows) != limit:
        raise RuntimeError(f"Expected {limit} samples, built {len(sample_rows)}")
    return sample_rows


def write_outputs(samples: list[dict], output_dir: Path, manifest_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "part-%06d.tar")

    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        with wds.ShardWriter(output_pattern, maxcount=SHARD_MAXCOUNT) as sink:
            for sample in samples:
                payload = {
                    "__key__": sample["sample_id"],
                    sample["image_ext"]: sample["image_bytes"],
                    "json": json.dumps(
                        {
                            **sample["metadata"],
                            "sample_id": sample["sample_id"],
                            "image_ext": sample["image_ext"],
                        },
                        ensure_ascii=False,
                    ).encode("utf-8"),
                }
                sink.write(payload)
                manifest_handle.write(
                    json.dumps(
                        {
                            "sample_id": sample["sample_id"],
                            **sample["metadata"],
                            "image_ext": sample["image_ext"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    args = parse_args()
    samples = build_sample_rows(limit=args.limit, seed=args.seed)
    write_outputs(samples, output_dir=args.output_dir, manifest_path=args.manifest_path)
    print(f"Built {len(samples)} WAFFLE samples into {args.output_dir} with manifest {args.manifest_path}")


if __name__ == "__main__":
    main()
