#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/UNO-1M")
IMAGE_DIR = DATASET_ROOT / "images"
CAPTIONS_BY_SPLIT_ROOT = Path(__file__).resolve().parent.parent / "cache" / "uno_1m_v3_by_split"
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/vision-datasets/UNO-1M___paired_recap_v3")
SHARD_MAXCOUNT = 10_000
SHARDS_PER_CHUNK = 5
PROGRESS_EVERY = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--captions-root", type=Path, default=CAPTIONS_BY_SPLIT_ROOT)
    return parser.parse_args()


def sample_key(sample_id: str) -> str:
    return str(Path(sample_id).with_suffix("")).replace("/", "__")


def image_ext(sample_id: str) -> str:
    ext = Path(sample_id).suffix.lower().lstrip(".")
    if not ext:
        raise ValueError(f"Missing image extension for {sample_id!r}")
    return ext


def iter_split_paths() -> list[Path]:
    paths = sorted(IMAGE_DIR.glob("split*.tar.gz"))
    if not paths:
        raise FileNotFoundError(f"No UNO image splits found under {IMAGE_DIR}")
    return paths


def assigned_splits(chunk_id: int, chunk_count: int) -> list[Path]:
    split_paths = iter_split_paths()
    total = len(split_paths)
    return split_paths[total * chunk_id // chunk_count : total * (chunk_id + 1) // chunk_count]


def load_split_captions(captions_root: Path, split_names: list[str]) -> dict[str, dict]:
    captions: dict[str, dict] = {}
    for split_name in split_names:
        path = captions_root / f"{split_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared caption file {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                sample_id = str(payload["sample_id"])
                metadata = dict(payload.get("metadata") or {})
                metadata.pop("original_caption", None)
                metadata["sample_id"] = sample_key(sample_id)
                metadata["image_path"] = sample_id
                metadata["caption_version"] = "uno_1m_v3"
                captions[sample_id] = {
                    "caption": payload["caption"],
                    "image_ext": image_ext(sample_id),
                    "metadata": metadata,
                }
    return captions


def build_payload(sample_id: str, image_bytes: bytes, caption_info: dict, source_tar: Path) -> dict[str, bytes]:
    metadata = {
        **caption_info["metadata"],
        "source_tar": str(source_tar),
    }
    return {
        "__key__": sample_key(sample_id),
        caption_info["image_ext"]: image_bytes,
        "txt": caption_info["caption"].encode("utf-8"),
        "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.chunk_id < args.chunk_count:
        raise ValueError(f"Invalid chunk {args.chunk_id}/{args.chunk_count}")

    splits = assigned_splits(args.chunk_id, args.chunk_count)
    split_names = [path.stem.removesuffix(".tar") for path in splits]
    print(
        f"Chunk {args.chunk_id}/{args.chunk_count - 1} assigned {len(splits)} splits: {split_names}",
        flush=True,
    )

    captions = load_split_captions(args.captions_root, split_names)
    print(
        f"Loaded {len(captions)} prepared captions for chunk {args.chunk_id}",
        flush=True,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_pattern = str(args.output_root / "part-%06d.tar")
    start_shard = args.chunk_id * SHARDS_PER_CHUNK
    written = 0
    seen: set[str] = set()

    with wds.ShardWriter(
        output_pattern,
        maxcount=SHARD_MAXCOUNT,
        start_shard=start_shard,
    ) as sink:
        for archive_path in splits:
            with tarfile.open(archive_path, "r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue

                    caption_info = captions.get(member.name)
                    if caption_info is None:
                        continue

                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Failed to extract {member.name!r} from {archive_path}")

                    sink.write(
                        build_payload(
                            sample_id=member.name,
                            image_bytes=extracted.read(),
                            caption_info=caption_info,
                            source_tar=archive_path,
                        )
                    )
                    seen.add(member.name)
                    written += 1
                    if written % PROGRESS_EVERY == 0:
                        print(
                            f"Chunk {args.chunk_id}: wrote {written} UNO samples",
                            flush=True,
                        )

    summary = {
        "chunk_id": args.chunk_id,
        "chunk_count": args.chunk_count,
        "assigned_splits": split_names,
        "captions_loaded": len(captions),
        "written": written,
        "unused_captions": len(captions) - len(seen),
        "start_shard": start_shard,
        "shards_per_chunk": SHARDS_PER_CHUNK,
    }
    summary_path = args.output_root / f"chunk_summary_{args.chunk_id:03d}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
