#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/openimages_v7_dense___full")
CAPTIONS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "openimages_v7_dense___full_v2"
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/vision-datasets/openimages_v7_dense___full_v2___recap")
IMAGE_KEYS_TO_EXTENSIONS = {
    "jpg": "jpg",
    "jpeg": "jpeg",
    "png": "png",
    "webp": "webp",
}
SHARD_MAXCOUNT = 2_500
PROGRESS_EVERY = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subdir", default=".")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def resolve_output_dir(output_root: Path, subdir: str) -> Path:
    if subdir in {"", "."}:
        return output_root
    return output_root / subdir


def normalize_caption(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").strip()


def load_captions() -> dict[str, str]:
    captions: dict[str, str] = {}
    caption_paths = sorted(CAPTIONS_ROOT.glob("captions_task*.jsonl"))
    if not caption_paths:
        raise FileNotFoundError(f"No caption files found under {CAPTIONS_ROOT}")

    loaded = 0
    for caption_path in caption_paths:
        with caption_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                sample_id = payload.get("sample_id")
                caption = normalize_caption(payload.get("caption"))
                if not sample_id or not caption:
                    continue
                captions[str(sample_id)] = caption
                loaded += 1
                if loaded % PROGRESS_EVERY == 0:
                    print(f"Loaded {loaded} captions", flush=True)

    print(f"Loaded {loaded} captions from {len(caption_paths)} files", flush=True)
    return captions


def iter_shard_paths() -> list[Path]:
    shard_paths = sorted(DATASET_ROOT.glob("*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found under {DATASET_ROOT}")
    return shard_paths


def extract_image(sample: dict) -> tuple[bytes, str]:
    for key, extension in IMAGE_KEYS_TO_EXTENSIONS.items():
        image_bytes = sample.get(key)
        if image_bytes is not None:
            return bytes(image_bytes), extension

    available_keys = ", ".join(sorted(sample))
    raise KeyError(f"No supported image payload found in sample keys: {available_keys}")


def iter_source_samples(limit: int | None):
    written = 0
    for shard_path in iter_shard_paths():
        dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)
        for sample in dataset:
            image_bytes, image_ext = extract_image(sample)
            metadata = json.loads(bytes(sample.get("json", b"{}")).decode("utf-8"))
            sample_id = sample["__key__"]
            metadata_key = metadata.get("__key__")
            if metadata_key is not None and metadata_key != sample_id:
                raise ValueError(f"Mismatched __key__ for {shard_path}: sample={sample_id!r} metadata={metadata_key!r}")

            yield {
                "sample_id": sample_id,
                "image_ext": image_ext,
                "image_bytes": image_bytes,
                "metadata": {
                    **metadata,
                    "sample_id": sample_id,
                    "source_shard": shard_path.name,
                    "source_tar": sample.get("__url__", str(shard_path)),
                },
            }
            written += 1
            if limit is not None and written >= limit:
                return


def build_sample_payload(source_sample: dict, caption: str) -> dict[str, bytes]:
    metadata = dict(source_sample["metadata"])
    metadata.pop("caption", None)
    return {
        "__key__": source_sample["sample_id"],
        source_sample["image_ext"]: source_sample["image_bytes"],
        "txt": caption.encode("utf-8"),
        "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
    }


def export_samples(
    output_dir: Path,
    captions: dict[str, str],
    limit: int | None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "part-%06d.tar")
    written = 0
    seen_caption_ids: set[str] = set()

    with wds.ShardWriter(output_pattern, maxcount=SHARD_MAXCOUNT) as sink:
        for source_sample in iter_source_samples(limit):
            sample_id = source_sample["sample_id"]
            caption = captions.get(sample_id)
            if caption is None:
                raise RuntimeError(f"Missing v2 caption for source sample {sample_id!r}")

            sink.write(build_sample_payload(source_sample, caption))
            seen_caption_ids.add(sample_id)
            written += 1
            if written % PROGRESS_EVERY == 0:
                if limit is None:
                    print(f"Wrote {written} OpenImages recap samples", flush=True)
                else:
                    print(f"Wrote {written}/{limit} OpenImages recap samples", flush=True)

    return {
        "written": written,
        "unused_captions": len(captions) - len(seen_caption_ids),
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_root, args.subdir)
    captions = load_captions()
    export_stats = export_samples(
        output_dir=output_dir,
        captions=captions,
        limit=args.limit,
    )

    summary = {
        "captions_loaded": len(captions),
        "limit": args.limit,
        "output_dir": str(output_dir),
        **export_stats,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
