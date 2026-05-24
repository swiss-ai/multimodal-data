#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from zipfile import ZipFile

import webdataset as wds

DATASET_ROOT = Path("/path/to/data/vision-datasets/MIT-10M")
CAPTIONS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "mit_10m"
DEFAULT_OUTPUT_ROOT = Path("/path/to/data/vision-datasets/MIT-10M___recap")
DEFAULT_IMAGE_TIER = "big"
TEXT_TEMPLATE = "lang=<code>\\n<caption> blocks separated by blank lines"
SHARD_MAXCOUNT = 10_000
PROGRESS_EVERY = 100_000
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--subdir", default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-tier", default=DEFAULT_IMAGE_TIER, choices=("big", "small"))
    return parser.parse_args()


def resolve_output_dir(output_root: Path, subdir: str) -> Path:
    if subdir in {"", "."}:
        return output_root
    return output_root / subdir


def normalize_caption(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def discover_languages() -> list[str]:
    languages = sorted(path.name for path in CAPTIONS_ROOT.iterdir() if path.is_dir() and not path.name.startswith("."))
    if "en" not in languages:
        raise FileNotFoundError(f"Missing English captions under {CAPTIONS_ROOT}")
    return languages


def iter_caption_payloads(language: str):
    language_dir = CAPTIONS_ROOT / language
    for caption_path in sorted(language_dir.glob("captions_task*.jsonl")):
        with caption_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                yield json.loads(line)


def extract_img_and_metadata(payload: dict) -> tuple[str, dict]:
    metadata = payload.get("metadata") or {}
    img = metadata.get("img")
    if not isinstance(img, str) or not img:
        raise ValueError(f"Missing img in payload: {payload!r}")

    image_archive_lang = metadata.get("image_archive_lang")
    if not isinstance(image_archive_lang, str) or not image_archive_lang:
        image_archive_lang = img.split("/", 1)[0]

    sample_metadata = {
        "img": img,
        "image_archive_lang": image_archive_lang,
        "box_cnt": metadata.get("box_cnt"),
        "difficulty": metadata.get("difficulty"),
        "cate_id": metadata.get("cate_id"),
        "cate_name": metadata.get("cate_name"),
        "split": metadata.get("split"),
    }
    return img, sample_metadata


def load_captions_into_memory(
    languages: list[str],
) -> tuple[dict[str, dict[str, str]], dict[str, dict], dict[str, int]]:
    captions_by_img: dict[str, dict[str, str]] = {}
    samples_by_img: dict[str, dict] = {}
    loaded_captions = 0

    for language in languages:
        for payload in iter_caption_payloads(language):
            caption = normalize_caption(payload.get("caption"))
            if not caption:
                continue

            img, sample_metadata = extract_img_and_metadata(payload)
            image_ext = Path(img).suffix.lower().lstrip(".")
            if not image_ext:
                continue

            captions_by_img.setdefault(img, {})[language] = caption
            if img not in samples_by_img:
                samples_by_img[img] = {
                    "img": img,
                    "image_archive_lang": sample_metadata["image_archive_lang"],
                    "image_ext": image_ext,
                    "metadata": sample_metadata,
                }

            loaded_captions += 1
            if loaded_captions % PROGRESS_EVERY == 0:
                print(f"Loaded {loaded_captions} MIT captions", flush=True)

    stats = {
        "captions_loaded": loaded_captions,
        "images_with_captions": len(samples_by_img),
    }
    return captions_by_img, samples_by_img, stats


def shuffled_languages(img: str, languages: list[str], seed: int) -> list[str]:
    other_languages = [language for language in languages if language != "en"]
    digest = hashlib.sha1(f"{img}:{seed}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], byteorder="big"))
    rng.shuffle(other_languages)
    return ["en", *other_languages]


def format_multilingual_text(
    img: str,
    captions: dict[str, str],
    languages: list[str],
    seed: int,
) -> tuple[str, list[str]]:
    ordered_languages = [language for language in shuffled_languages(img, languages, seed) if language in captions]
    sections = [f"lang={language}\n{captions[language]}" for language in ordered_languages]
    return "\n\n".join(sections), ordered_languages


def sample_key_for_img(img: str) -> str:
    return str(Path(img).with_suffix("")).replace("/", "__")


def build_sample_payload(
    source_sample: dict,
    image_bytes: bytes,
    captions: dict[str, str],
    languages: list[str],
    seed: int,
    image_tier: str,
) -> dict[str, bytes]:
    img = source_sample["img"]
    sample_key = sample_key_for_img(img)
    text, ordered_languages = format_multilingual_text(
        img=img,
        captions=captions,
        languages=languages,
        seed=seed,
    )
    metadata = {
        **source_sample["metadata"],
        "sample_id": sample_key,
        "available_languages": sorted(captions),
        "image_tier": image_tier,
        "language_order": ordered_languages,
        "text_template": TEXT_TEMPLATE,
    }
    return {
        "__key__": sample_key,
        source_sample["image_ext"]: image_bytes,
        "txt": text.encode("utf-8"),
        "json": json.dumps(metadata, ensure_ascii=False).encode("utf-8"),
    }


def iter_source_images(
    image_tier: str,
    captions_by_img: dict[str, dict[str, str]],
    samples_by_img: dict[str, dict],
    limit: int | None,
):
    written = 0
    data_dir = DATASET_ROOT / "data" / image_tier
    archive_paths = sorted(data_dir.glob("*.zip"))
    if not archive_paths:
        raise FileNotFoundError(f"No image archives found under {data_dir}")

    for archive_path in archive_paths:
        with ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue

                img = info.filename
                if not img.lower().endswith(IMAGE_SUFFIXES):
                    continue

                source_sample = samples_by_img.get(img)
                if source_sample is None:
                    continue

                captions = captions_by_img.get(img, {})
                if "en" not in captions:
                    continue

                yield source_sample, captions, archive.read(info)
                written += 1
                if limit is not None and written >= limit:
                    return


def export_webdataset(
    source_iter,
    languages: list[str],
    output_dir: Path,
    seed: int,
    limit: int | None,
    image_tier: str,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "part-%06d.tar")
    written = 0

    with wds.ShardWriter(output_pattern, maxcount=SHARD_MAXCOUNT) as sink:
        for source_sample, captions, image_bytes in source_iter:
            sink.write(
                build_sample_payload(
                    source_sample=source_sample,
                    image_bytes=image_bytes,
                    captions=captions,
                    languages=languages,
                    seed=seed,
                    image_tier=image_tier,
                )
            )
            written += 1
            if written % PROGRESS_EVERY == 0:
                if limit is None:
                    print(f"Wrote {written} MIT samples", flush=True)
                else:
                    print(f"Wrote {written}/{limit} MIT samples", flush=True)

    return {"written": written}


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_root, args.subdir)
    languages = discover_languages()
    captions_by_img, samples_by_img, load_stats = load_captions_into_memory(languages)
    export_stats = export_webdataset(
        source_iter=iter_source_images(
            image_tier=args.image_tier,
            captions_by_img=captions_by_img,
            samples_by_img=samples_by_img,
            limit=args.limit,
        ),
        languages=languages,
        output_dir=output_dir,
        seed=args.seed,
        limit=args.limit,
        image_tier=args.image_tier,
    )

    summary = {
        "image_tier": args.image_tier,
        "languages": languages,
        "limit": args.limit,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "text_template": TEXT_TEMPLATE,
        **load_stats,
        **export_stats,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
