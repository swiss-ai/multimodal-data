#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import webdataset as wds

WORKDIR = Path(__file__).resolve().parent.parent
_LOCAL_ROOT = WORKDIR / "outputs" / "tau_waffle_architecture_gemma4_sample16"
SAMPLE_ROOT = _LOCAL_ROOT / "sample_webdataset"
CAPSTOR_ROOT = Path("/tmp/test")
CAPTIONS_ROOT = CAPSTOR_ROOT / "captions"
OUTPUT_ROOT = CAPSTOR_ROOT / "inspection"
IMAGE_KEYS = ("jpg", "jpeg", "png", "webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, default=SAMPLE_ROOT)
    parser.add_argument("--captions-root", type=Path, default=CAPTIONS_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_captions(captions_root: Path) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    caption_paths = sorted(captions_root.glob("captions_task*.jsonl"))
    if not caption_paths:
        raise FileNotFoundError(f"No caption files found under {captions_root}")

    for caption_path in caption_paths:
        with caption_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                payloads[payload["sample_id"]] = payload
    return payloads


def parse_sections(text: str) -> tuple[str | None, str]:
    marker_perspective = "### Perspective"
    marker_caption = "### Caption"
    if marker_perspective not in text or marker_caption not in text:
        return None, text.strip()

    _, after_perspective = text.split(marker_perspective, 1)
    perspective_text, caption_text = after_perspective.split(marker_caption, 1)
    return perspective_text.strip(), caption_text.strip()


def clean_generated_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace("<turn|>", "").strip()
    return cleaned


def iter_samples(sample_root: Path):
    shard_paths = sorted(sample_root.glob("*.tar"))
    if not shard_paths:
        raise FileNotFoundError(f"No sample shards found under {sample_root}")
    for shard_path in shard_paths:
        dataset = wds.WebDataset(str(shard_path), shardshuffle=False, empty_check=False)
        for sample in dataset:
            yield sample


def export_sample(
    sample: dict,
    caption_payload: dict,
    output_root: Path,
    overwrite: bool,
) -> None:
    metadata = json.loads(bytes(sample["json"]).decode("utf-8"))
    sample_id = metadata["sample_id"]
    sample_dir = output_root / sample_id
    if sample_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{sample_dir} already exists; use --overwrite")
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_ext = metadata["image_ext"]
    image_bytes = bytes(sample[image_ext])
    image_path = sample_dir / f"image.{image_ext}"
    image_path.write_bytes(image_bytes)

    raw_response = caption_payload["caption"].strip()
    perspective, caption_text = parse_sections(raw_response)
    perspective = clean_generated_text(perspective)
    caption_text = clean_generated_text(caption_text) or ""
    (sample_dir / "caption.txt").write_text(caption_text + "\n", encoding="utf-8")

    caption_metadata = dict(caption_payload.get("metadata") or {})
    inspection_metadata = {
        **metadata,
        **caption_metadata,
        "model_used": caption_metadata.get("model_used"),
        "model_repo": caption_metadata.get("model_repo"),
        "thinking_enabled": caption_metadata.get("thinking_enabled"),
        "raw_response": raw_response,
        "perspective": perspective,
        "caption_text": caption_text,
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(inspection_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    captions = load_captions(args.captions_root)
    samples = list(iter_samples(args.sample_root))

    if len(samples) != 16:
        raise RuntimeError(f"Expected 16 samples, found {len(samples)}")
    if len(captions) != 16:
        raise RuntimeError(f"Expected 16 captions, found {len(captions)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    for sample in samples:
        metadata = json.loads(bytes(sample["json"]).decode("utf-8"))
        sample_id = metadata["sample_id"]
        caption_payload = captions.get(sample_id)
        if caption_payload is None:
            raise KeyError(f"Missing caption for sample_id={sample_id}")
        export_sample(sample, caption_payload, args.output_root, overwrite=args.overwrite)

    print(f"Wrote {len(samples)} inspection directories to {args.output_root}")


if __name__ == "__main__":
    main()
