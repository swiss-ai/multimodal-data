#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from hashlib import blake2b
from pathlib import Path

DATASET_ROOT = Path("/path/to/data/vision-datasets/UNO-1M")
LABEL_DIR = DATASET_ROOT / "labels"
IMAGE_DIR = DATASET_ROOT / "images"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "manifests" / "uno_1m_stage2"
TASK_COUNT = 64
FILTERS = {
    "hq35": 3.5,
    "hq40": 4.0,
}


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def normalize_subjects(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    subjects = []
    seen = set()
    for item in value:
        text = normalize_text(item)
        key = text.casefold()
        if text and key not in seen:
            subjects.append(text)
            seen.add(key)
    return subjects


def sentence_without_terminal_punct(text: str) -> str:
    return text.rstrip().rstrip(".!? ")


def lowercase_first(text: str) -> str:
    if not text:
        return text
    return text[:1].lower() + text[1:]


def strip_leading_article(text: str) -> str:
    lowered = text.casefold()
    for prefix in ("a ", "an ", "the "):
        if lowered.startswith(prefix):
            return text[len(prefix) :]
    return text


def anchor_subject(subjects: list[str]) -> str:
    if not subjects:
        return ""
    anchor = normalize_text(subjects[0])
    if "," in anchor and " and " not in anchor.casefold():
        anchor = normalize_text(anchor.split(",", 1)[0])
    anchor = strip_leading_article(anchor)
    return lowercase_first(anchor)


def stable_task_id(sample_id: str) -> int:
    digest = blake2b(sample_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % TASK_COUNT


def build_pair_text(caption1: str, caption2: str, subjects: list[str]) -> str:
    caption1 = normalize_text(caption1)
    caption2 = normalize_text(caption2)
    if not caption1:
        return caption2
    if not caption2:
        return caption1
    if caption1 == caption2:
        return caption1

    second_view = lowercase_first(sentence_without_terminal_punct(caption2))
    subject_anchor = anchor_subject(subjects)
    if subject_anchor:
        return (
            f"{caption1} "
            f"A matching variant keeps the same {subject_anchor} "
            f"but changes the setting or presentation: {second_view}."
        )

    return (
        f"{caption1} A matching variant keeps the main subject but changes the setting or presentation: {second_view}."
    )


def iter_label_files(limit_splits: int | None) -> list[Path]:
    files = sorted(LABEL_DIR.glob("split*.json"))
    if limit_splits is not None:
        return files[:limit_splits]
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-splits", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for filter_name in FILTERS:
        filter_dir = output_root / filter_name
        filter_dir.mkdir(parents=True, exist_ok=True)

    stats = {filter_name: {"written": 0, "splits": 0} for filter_name in FILTERS}

    with ExitStack() as stack:
        handles = {}
        for filter_name in FILTERS:
            for task_id in range(TASK_COUNT):
                path = output_root / filter_name / f"manifest_task{task_id:04d}.jsonl"
                handles[(filter_name, task_id)] = stack.enter_context(path.open("w", encoding="utf-8"))

        for label_path in iter_label_files(args.limit_splits):
            entries = json.loads(label_path.read_text(encoding="utf-8"))
            split_name = label_path.stem
            for filter_name in FILTERS:
                stats[filter_name]["splits"] += 1

            for row_index, row in enumerate(entries):
                caption = row.get("caption") or {}
                caption1 = normalize_text(caption.get("img_path1"))
                caption2 = normalize_text(caption.get("img_path2"))
                subjects = normalize_subjects(caption.get("subject"))
                score_final = float((row.get("vlm_filter_cot") or {}).get("score_final", 0.0))

                sample_id = f"{row['img_path1']}||{row['img_path2']}"
                task_id = stable_task_id(sample_id)
                payload = {
                    "sample_id": sample_id,
                    "split": split_name,
                    "archive_path": str(IMAGE_DIR / f"{split_name}.tar.gz"),
                    "img_path1": row["img_path1"],
                    "img_path2": row["img_path2"],
                    "caption1": caption1,
                    "caption2": caption2,
                    "pair_text": build_pair_text(caption1, caption2, subjects),
                    "subject": subjects,
                    "judgment": normalize_text(caption.get("judgment")),
                    "score_final": score_final,
                    "score_part": (row.get("vlm_filter_cot") or {}).get("score_part") or {},
                    "row_index": row_index,
                }

                for filter_name, threshold in FILTERS.items():
                    if score_final < threshold:
                        continue
                    handles[(filter_name, task_id)].write(json.dumps(payload, ensure_ascii=False) + "\n")
                    stats[filter_name]["written"] += 1

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
