#!/usr/bin/env python3
"""Merge grounded captions into flat output tars: {key}.jpg + {key}.json per image."""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

WORK_DIR = Path(os.environ.get("RECAPTION_WORK_DIR", "/tmp/recaption_blip3"))
VALID_DIR = WORK_DIR / "output_valid"
OUTPUT_ROOT = Path(os.environ.get("RECAPTION_MERGE_OUTPUT_ROOT", ""))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-map", type=str, default=str(WORK_DIR / "chunk_map_800.json"))
    args = parser.parse_args()

    with open(args.chunk_map) as f:
        chunk_map = json.load(f)

    tar_paths = chunk_map.get(str(args.chunk_id), [])
    if not tar_paths:
        print(f"Chunk {args.chunk_id}: no tars assigned, skipping.")
        return

    jsonl_path = VALID_DIR / f"{args.chunk_id:04d}.jsonl"
    if not jsonl_path.exists():
        print(f"Chunk {args.chunk_id}: no valid JSONL found, skipping.")
        return

    # Load captions for this chunk, keyed by (key, source_tar)
    index: dict[tuple[str, str], str] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            index[(r["key"], r["source_tar"])] = r["caption"]

    print(
        f"Chunk {args.chunk_id}: {len(index)} captions, {len(tar_paths)} tars",
        flush=True,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for tar_idx, tar_path in enumerate(tar_paths):
        out_path = OUTPUT_ROOT / f"{args.chunk_id:04d}_{tar_idx:02d}.tar"

        if out_path.exists():
            print(f"  {out_path.name}: already exists, skipping", flush=True)
            continue

        written = 0
        try:
            with (
                tarfile.open(tar_path, "r") as tf_in,
                tarfile.open(str(out_path) + ".tmp", "w") as tf_out,
            ):
                name_to_member = {m.name: m for m in tf_in.getmembers()}
                jpg_members = sorted(
                    (m for m in name_to_member.values() if m.name.endswith(".jpg")),
                    key=lambda m: m.name,
                )

                for jpg_member in jpg_members:
                    key = jpg_member.name[:-4]
                    caption = index.get((key, tar_path))
                    if caption is None:
                        continue

                    # Write jpg
                    tf_out.addfile(jpg_member, tf_in.extractfile(jpg_member))

                    # Write json with caption and provenance
                    payload = json.dumps(
                        {
                            "caption": caption,
                            "source_tar": tar_path,
                            "key": key,
                        },
                        ensure_ascii=False,
                    ).encode("utf-8")
                    info = tarfile.TarInfo(name=f"{key}.json")
                    info.size = len(payload)
                    tf_out.addfile(info, io.BytesIO(payload))
                    written += 1

        except Exception as e:
            Path(str(out_path) + ".tmp").unlink(missing_ok=True)
            print(f"  error on {tar_path}: {e}", flush=True)
            continue

        Path(str(out_path) + ".tmp").rename(out_path)
        total_written += written
        print(f"  {out_path.name}: {written} images", flush=True)

    print(f"Chunk {args.chunk_id}: done, {total_written} total images written", flush=True)


if __name__ == "__main__":
    main()
