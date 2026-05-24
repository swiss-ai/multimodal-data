#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/UNO-1M")
IMAGE_DIR = DATASET_ROOT / "images"
CAPTIONS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "uno_1m_v3"
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/vision-datasets/UNO-1M___paired_recap_v3")
SHARD_MAXCOUNT = 10_000
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


def sample_key(sample_id: str) -> str:
    return str(Path(sample_id).with_suffix("")).replace("/", "__")


def image_ext(sample_id: str) -> str:
    ext = Path(sample_id).suffix.lower().lstrip(".")
    if not ext:
        raise ValueError(f"Missing image extension for {sample_id!r}")
    return ext


def load_captions() -> dict[str, dict]:
    caption_paths = sorted(CAPTIONS_ROOT.glob("captions_task*.jsonl"))
    if not caption_paths:
        raise FileNotFoundError(f"No UNO v3 caption files found under {CAPTIONS_ROOT}")

    captions: dict[str, dict] = {}
    loaded = 0
    for caption_path in caption_paths:
        with caption_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                sample_id = payload.get("sample_id")
                caption = normalize_caption(payload.get("caption"))
                metadata = dict(payload.get("metadata") or {})
                if not sample_id or not caption:
                    continue

                metadata.pop("original_caption", None)
                metadata["sample_id"] = sample_key(str(sample_id))
                metadata["image_path"] = sample_id
                metadata["caption_version"] = "uno_1m_v3"
                captions[str(sample_id)] = {
                    "caption": caption,
                    "image_ext": image_ext(str(sample_id)),
                    "metadata": metadata,
                }
                loaded += 1
                if loaded % PROGRESS_EVERY == 0:
                    print(f"Loaded {loaded} UNO captions", flush=True)

    print(f"Loaded {loaded} UNO captions from {len(caption_paths)} files", flush=True)
    return captions


def iter_archive_paths() -> list[Path]:
    paths = sorted(IMAGE_DIR.glob("split*.tar.gz"))
    if not paths:
        raise FileNotFoundError(f"No UNO image splits found under {IMAGE_DIR}")
    return paths


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


def export_samples(
    output_dir: Path,
    captions: dict[str, dict],
    limit: int | None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "part-%06d.tar")
    written = 0
    seen: set[str] = set()

    with wds.ShardWriter(output_pattern, maxcount=SHARD_MAXCOUNT) as sink:
        for archive_path in iter_archive_paths():
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
                        if limit is None:
                            print(f"Wrote {written} UNO samples", flush=True)
                        else:
                            print(f"Wrote {written}/{limit} UNO samples", flush=True)
                    if limit is not None and written >= limit:
                        return {
                            "written": written,
                            "unused_captions": len(captions) - len(seen),
                        }

    return {
        "written": written,
        "unused_captions": len(captions) - len(seen),
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
