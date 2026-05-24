#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/UNO-1M")
CAPTIONS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "uno_1m_v3"
LABEL_DIR = DATASET_ROOT / "labels"
IMAGE_DIR = DATASET_ROOT / "images"
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/vision-datasets/UNO-1M___paired_recap_v3")
TEXT_TEMPLATE = "view=1/view=2 blocks separated by blank lines"
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


def iter_label_paths() -> list[Path]:
    paths = sorted(LABEL_DIR.glob("split*.json"))
    if not paths:
        raise FileNotFoundError(f"No UNO label splits found under {LABEL_DIR}")
    return paths


def load_captions() -> dict[str, str]:
    captions: dict[str, str] = {}
    loaded = 0
    caption_paths = sorted(CAPTIONS_ROOT.glob("captions_task*.jsonl"))
    if not caption_paths:
        raise FileNotFoundError(f"No UNO v3 caption files found under {CAPTIONS_ROOT}")

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
                    print(f"Loaded {loaded} UNO captions", flush=True)

    print(f"Loaded {loaded} UNO captions from {len(caption_paths)} files", flush=True)
    return captions


def sample_key(split_name: str, row_index: int) -> str:
    return f"{split_name}__row{row_index:07d}"


def image_ext(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".")
    if not ext:
        raise ValueError(f"Missing image extension for {image_path!r}")
    return ext


def pair_text(caption1: str, caption2: str) -> str:
    return f"view=1\n{caption1}\n\nview=2\n{caption2}"


def build_metadata(split_name: str, row_index: int, row: dict) -> dict:
    caption = row.get("caption") or {}
    vlm_filter = row.get("vlm_filter_cot") or {}
    return {
        "sample_id": sample_key(split_name, row_index),
        "pair_id": f"{row['img_path1']}||{row['img_path2']}",
        "split": split_name,
        "row_index": row_index,
        "archive_path": str(IMAGE_DIR / f"{split_name}.tar.gz"),
        "img_path1": row["img_path1"],
        "img_path2": row["img_path2"],
        "subject": caption.get("subject") or [],
        "judgment": caption.get("judgment"),
        "original_caption1": caption.get("img_path1"),
        "original_caption2": caption.get("img_path2"),
        "score_final": vlm_filter.get("score_final"),
        "score_part": vlm_filter.get("score_part") or {},
        "caption_version": "uno_1m_v3",
        "text_template": TEXT_TEMPLATE,
        "image_fields": [
            f"view1.{image_ext(row['img_path1'])}",
            f"view2.{image_ext(row['img_path2'])}",
        ],
        "text_fields": ["txt", "view1.txt", "view2.txt"],
    }


def build_payload(
    split_name: str,
    row_index: int,
    row: dict,
    image_bytes1: bytes,
    image_bytes2: bytes,
    caption1: str,
    caption2: str,
) -> dict[str, bytes]:
    return {
        "__key__": sample_key(split_name, row_index),
        f"view1.{image_ext(row['img_path1'])}": image_bytes1,
        f"view2.{image_ext(row['img_path2'])}": image_bytes2,
        "view1.txt": caption1.encode("utf-8"),
        "view2.txt": caption2.encode("utf-8"),
        "txt": pair_text(caption1, caption2).encode("utf-8"),
        "json": json.dumps(
            build_metadata(split_name, row_index, row),
            ensure_ascii=False,
        ).encode("utf-8"),
    }


def export_pairs(
    output_dir: Path,
    captions: dict[str, str],
    limit: int | None,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "part-%06d.tar")
    written = 0
    missing_captions = 0

    with wds.ShardWriter(output_pattern, maxcount=SHARD_MAXCOUNT) as sink:
        for label_path in iter_label_paths():
            split_name = label_path.stem
            archive_path = IMAGE_DIR / f"{split_name}.tar.gz"
            rows = json.loads(label_path.read_text(encoding="utf-8"))
            requested_images: dict[str, list[tuple[int, int]]] = {}
            pair_states: dict[int, dict] = {}

            for row_index, row in enumerate(rows):
                if limit is not None and written + len(pair_states) >= limit:
                    break

                caption1 = captions.get(row["img_path1"])
                caption2 = captions.get(row["img_path2"])
                if not caption1 or not caption2:
                    missing_captions += 1
                    continue

                pair_states[row_index] = {
                    "row": row,
                    "caption1": caption1,
                    "caption2": caption2,
                    "image_bytes1": None,
                    "image_bytes2": None,
                }
                requested_images.setdefault(row["img_path1"], []).append((row_index, 1))
                requested_images.setdefault(row["img_path2"], []).append((row_index, 2))

            if not pair_states:
                if limit is not None and written >= limit:
                    break
                continue

            with tarfile.open(archive_path, "r|gz") as archive:
                for member in archive:
                    if not member.isfile():
                        continue

                    hits = requested_images.pop(member.name, None)
                    if not hits:
                        continue

                    extracted = archive.extractfile(member)
                    if extracted is None:
                        raise RuntimeError(f"Failed to extract {member.name!r} from {archive_path}")
                    image_bytes = extracted.read()

                    for row_index, side in hits:
                        state = pair_states.get(row_index)
                        if state is not None:
                            state[f"image_bytes{side}"] = image_bytes

                    if not requested_images:
                        break

            if requested_images:
                missing = ", ".join(sorted(requested_images)[:5])
                raise FileNotFoundError(
                    f"Missing {len(requested_images)} requested UNO images in {archive_path}; first missing: {missing}"
                )

            for row_index in sorted(pair_states):
                state = pair_states[row_index]
                image_bytes1 = state["image_bytes1"]
                image_bytes2 = state["image_bytes2"]
                if image_bytes1 is None or image_bytes2 is None:
                    raise RuntimeError(f"Incomplete UNO pair state for {split_name} row {row_index}")

                sink.write(
                    build_payload(
                        split_name=split_name,
                        row_index=row_index,
                        row=state["row"],
                        image_bytes1=image_bytes1,
                        image_bytes2=image_bytes2,
                        caption1=state["caption1"],
                        caption2=state["caption2"],
                    )
                )
                written += 1
                if written % PROGRESS_EVERY == 0:
                    if limit is None:
                        print(f"Wrote {written} UNO pairs", flush=True)
                    else:
                        print(f"Wrote {written}/{limit} UNO pairs", flush=True)
                if limit is not None and written >= limit:
                    return {
                        "written": written,
                        "missing_caption_pairs": missing_captions,
                    }

    return {
        "written": written,
        "missing_caption_pairs": missing_captions,
    }


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_root, args.subdir)
    captions = load_captions()
    export_stats = export_pairs(
        output_dir=output_dir,
        captions=captions,
        limit=args.limit,
    )

    summary = {
        "captions_loaded": len(captions),
        "limit": args.limit,
        "output_dir": str(output_dir),
        "text_template": TEXT_TEMPLATE,
        **export_stats,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
