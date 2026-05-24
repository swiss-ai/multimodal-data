#!/usr/bin/env python3

from __future__ import annotations

import json
from contextlib import ExitStack
from hashlib import blake2b
from pathlib import Path

DATASET_ROOT = Path("/path/to/data/vision-datasets/MIT-10M")
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "manifests" / "mit_10m_big"
TASK_COUNT = 64
LANGS = [
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "ja",
    "ko",
    "pt",
    "ru",
    "th",
    "tr",
    "zh",
]
SOURCE_FILES = [
    TRAIN_DIR / "DE.jsonl",
    TRAIN_DIR / "EN.jsonl",
    TRAIN_DIR / "ES.jsonl",
    TRAIN_DIR / "FR.jsonl",
    TRAIN_DIR / "IT.jsonl",
    TRAIN_DIR / "JA.jsonl",
    TRAIN_DIR / "PT.jsonl",
    TRAIN_DIR / "ZH.jsonl",
    TEST_DIR / "DE.jsonl",
    TEST_DIR / "EN.jsonl",
    TEST_DIR / "ES.jsonl",
    TEST_DIR / "FR.jsonl",
    TEST_DIR / "IT.jsonl",
    TEST_DIR / "JA.jsonl",
    TEST_DIR / "PT.jsonl",
    TEST_DIR / "ZH.jsonl",
]


def normalize_lang(value: str) -> str:
    return value.strip().lower()


def stable_task_id(sample_id: str) -> int:
    digest = blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % TASK_COUNT


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for lang in LANGS:
        lang_dir = OUTPUT_ROOT / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        handles: dict[tuple[str, int], object] = {}
        for lang in LANGS:
            for task_id in range(TASK_COUNT):
                path = OUTPUT_ROOT / lang / f"manifest_task{task_id:04d}.jsonl"
                handles[(lang, task_id)] = stack.enter_context(path.open("w", encoding="utf-8"))

        stats = {
            "written": {lang: 0 for lang in LANGS},
            "duplicates_skipped": {lang: 0 for lang in LANGS},
        }

        for source_path in SOURCE_FILES:
            source_lang = source_path.stem
            seen_source_images: set[str] = set()
            with source_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = json.loads(line)
                    img = row["img"]
                    src_lang = normalize_lang(row["src_lang"])
                    tgt_lang = normalize_lang(row["tgt_lang"])
                    if src_lang != normalize_lang(source_lang):
                        raise ValueError(f"Unexpected src_lang in {source_path}:{line_number}: {src_lang}")

                    if img not in seen_source_images:
                        seen_source_images.add(img)
                        sample_id = f"{img}::{src_lang}"
                        task_id = stable_task_id(sample_id)
                        payload = {
                            "sample_id": sample_id,
                            "img": img,
                            "image_archive_lang": img.split("/", 1)[0],
                            "conditioning_lang": src_lang,
                            "conditioning_text": row["src_text"],
                            "text_role": "src",
                            "box_cnt": row["box_cnt"],
                            "difficulty": row["difficulty"],
                            "cate_id": row["cate_id"],
                            "cate_name": row["cate_name"],
                            "split": row["split"],
                            "row_id": row["id"],
                        }
                        handles[(src_lang, task_id)].write(json.dumps(payload, ensure_ascii=False) + "\n")
                        stats["written"][src_lang] += 1
                    else:
                        stats["duplicates_skipped"][src_lang] += 1

                    sample_id = f"{img}::{tgt_lang}"
                    task_id = stable_task_id(sample_id)
                    payload = {
                        "sample_id": sample_id,
                        "img": img,
                        "image_archive_lang": img.split("/", 1)[0],
                        "conditioning_lang": tgt_lang,
                        "conditioning_text": row["tgt_text"],
                        "text_role": "tgt",
                        "box_cnt": row["box_cnt"],
                        "difficulty": row["difficulty"],
                        "cate_id": row["cate_id"],
                        "cate_name": row["cate_name"],
                        "split": row["split"],
                        "row_id": row["id"],
                    }
                    handles[(tgt_lang, task_id)].write(json.dumps(payload, ensure_ascii=False) + "\n")
                    stats["written"][tgt_lang] += 1

    for lang in LANGS:
        print(f"{lang}: written={stats['written'][lang]} source_dupes_skipped={stats['duplicates_skipped'][lang]}")


if __name__ == "__main__":
    main()
