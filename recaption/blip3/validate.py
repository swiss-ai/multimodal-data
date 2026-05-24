#!/usr/bin/env python3
"""Validate grounded captions: check object/bbox structure.

Rules:
  - No object/bbox tags at all → valid, keep as-is
  - Bbox before first object → valid, keep as-is
  - Object with no following bbox → remove that <object>tag</object>, keep rest
  - Object with bbox but invalid coords → drop record, log to output_invalid/

Writes:
  output_valid/{chunk_id:04d}.jsonl   — clean/fixed records
  output_invalid/{chunk_id:04d}.jsonl — dropped records with reason + key/source_tar
"""

import argparse
import json
import os
import re
from pathlib import Path

WORK_DIR = Path(os.environ.get("RECAPTION_WORK_DIR", "/tmp/recaption_blip3"))
OUTPUT_DIR = WORK_DIR / "output"
VALID_DIR = WORK_DIR / "output_valid"
INVALID_DIR = WORK_DIR / "output_invalid"

_OBJECT_RE = re.compile(r"<object>([^<]*)</object>")
_BBOX_COORDS_RE = re.compile(r"<bbox>\[([^\]]+)\]")

# Trailing fragments from token-limit cutoff
_TRAILING_OBJECT_RE = re.compile(r"\s*<object>[^<]*(?:</object)?$")  # <object>X or <object>X</object (no >)
_TRAILING_BBOX_RE = re.compile(r"\s*<bbox>[^\[]*(?:\[[^\]]*)?$")  # <bbox> or <bbox>[partial


def strip_truncation(caption: str) -> str:
    caption = _TRAILING_OBJECT_RE.sub("", caption)
    caption = _TRAILING_BBOX_RE.sub("", caption)
    return caption.rstrip()


def trim_incomplete_sentence(caption: str) -> str:
    last = caption.rstrip()
    if last and last[-1] not in {".", "\n", ">"}:
        dot = caption.rfind(".")
        caption = caption[: dot + 1] if dot != -1 else ""
    return caption


def process_caption(caption: str) -> tuple[str | None, str]:
    """Return (fixed_caption, reason). fixed_caption is None if record should be dropped."""
    caption = strip_truncation(caption)
    caption = trim_incomplete_sentence(caption)

    if len(caption.strip()) < 100:
        return None, "caption too short after trimming"

    objects = list(_OBJECT_RE.finditer(caption))

    # No structured tags at all — fine
    if not objects:
        return caption, ""

    to_remove = []
    for i, obj in enumerate(objects):
        label = obj.group(1)
        seg_end = objects[i + 1].start() if i + 1 < len(objects) else len(caption)
        segment = caption[obj.end() : seg_end]

        bbox_list = _BBOX_COORDS_RE.findall(segment)
        if not bbox_list:
            to_remove.append((obj.start(), obj.end()))
            continue

        for coords_str in bbox_list:
            try:
                coords = [float(x.strip()) for x in coords_str.split(",")]
            except ValueError:
                return None, f"non-numeric coords for '{label}'"
            if len(coords) != 4:
                return None, f"expected 4 coords, got {len(coords)} for '{label}'"
            x0, y0, x1, y1 = coords
            if not all(0 <= c <= 1000 for c in coords):
                return None, f"coords out of [0,1000] for '{label}'"
            if x0 > x1 or y0 > y1:
                return None, f"degenerate box for '{label}'"

    if to_remove:
        result = []
        prev = 0
        for start, end in to_remove:
            result.append(caption[prev:start])
            prev = end
        result.append(caption[prev:])
        caption = trim_incomplete_sentence("".join(result))
        if len(caption.strip()) < 100:
            return None, "caption too short after trimming"

    return caption, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, required=True)
    args = parser.parse_args()

    src = OUTPUT_DIR / f"{args.chunk_id:04d}.jsonl"
    if not src.exists():
        print(f"Chunk {args.chunk_id}: no output file, skipping.")
        return

    VALID_DIR.mkdir(parents=True, exist_ok=True)
    INVALID_DIR.mkdir(parents=True, exist_ok=True)
    valid_dst = VALID_DIR / f"{args.chunk_id:04d}.jsonl"
    invalid_dst = INVALID_DIR / f"{args.chunk_id:04d}.jsonl"

    total = valid = fixed = dropped = 0
    reasons: dict[str, int] = {}

    with (
        open(src) as f_in,
        open(valid_dst, "w") as f_valid,
        open(invalid_dst, "w") as f_invalid,
    ):
        for line_number, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                entry = {
                    "chunk_id": args.chunk_id,
                    "line_number": line_number,
                    "key": None,
                    "source_tar": None,
                    "invalid_reason": "malformed json",
                }
                f_invalid.write(json.dumps(entry, ensure_ascii=False) + "\n")
                reasons["malformed json"] = reasons.get("malformed json", 0) + 1
                dropped += 1
                continue

            new_caption, reason = process_caption(record.get("caption", ""))
            if new_caption is None:
                entry = {
                    "chunk_id": args.chunk_id,
                    "line_number": line_number,
                    "key": record.get("key"),
                    "source_tar": record.get("source_tar"),
                    "invalid_reason": reason,
                }
                f_invalid.write(json.dumps(entry, ensure_ascii=False) + "\n")
                reasons[reason] = reasons.get(reason, 0) + 1
                dropped += 1
            else:
                if new_caption != record["caption"]:
                    record = {**record, "caption": new_caption}
                    fixed += 1
                f_valid.write(json.dumps(record, ensure_ascii=False) + "\n")
                valid += 1

    print(
        json.dumps(
            {
                "chunk": args.chunk_id,
                "total": total,
                "valid": valid,
                "fixed": fixed,
                "dropped": dropped,
                "reasons": reasons,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
