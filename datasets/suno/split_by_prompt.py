#!/usr/bin/env python3
"""Split Suno clips by metadata_gpt_description_prompt, create sidecars, and export metadata.

For each clip, writes sidecar files alongside the .mp3:
  - {id}.url  — audio_url (always)
  - {id}.txt  — metadata_gpt_description_prompt (only if non-empty)

File lists include all files per clip (mp3 + sidecars) for tar packing.
"""

import argparse
import json
import os

import polars as pl


METADATA_COLUMNS = [
    "id", "audio_url", "title", "play_count",
    "metadata_tags", "metadata_prompt", "metadata_duration",
    "metadata_gpt_description_prompt",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = (
        pl.scan_parquet(args.parquet)
        .select(METADATA_COLUMNS)
        .unique(subset="id", keep="first")
        .collect()
    )

    on_disk = set(os.listdir(args.clips_dir))
    df = df.filter(pl.col("id").map_elements(lambda sid: f"{sid}.mp3" in on_disk, return_dtype=pl.Boolean))

    has_prompt = pl.col("metadata_gpt_description_prompt").fill_null("").str.strip_chars().str.len_chars() > 0
    wp = df.filter(has_prompt)
    np_ = df.filter(~has_prompt)

    for subset, name in [(wp, "with_prompt"), (np_, "without_prompt")]:
        file_list = []
        for row in subset.select(METADATA_COLUMNS).iter_rows(named=True):
            clip_id = row["id"]
            base = os.path.join(args.clips_dir, clip_id)

            # Always include audio
            file_list.append(f"{base}.mp3")

            # Write .url sidecar
            url = (row.get("audio_url") or "").strip()
            if url:
                url_path = f"{base}.url"
                with open(url_path, "w") as f:
                    f.write(url)
                file_list.append(url_path)

            # Write .txt sidecar (gpt description prompt) only for with_prompt
            text = (row.get("metadata_gpt_description_prompt") or "").strip()
            if text:
                txt_path = f"{base}.txt"
                with open(txt_path, "w") as f:
                    f.write(text)
                file_list.append(txt_path)

        # Write file list (all files per clip grouped together)
        with open(f"{args.output_dir}/files_{name}.txt", "w") as f:
            f.write("\n".join(file_list) + "\n")

        # Write metadata JSONL
        with open(f"{args.output_dir}/metadata_{name}.jsonl", "w") as f:
            for row in subset.select(METADATA_COLUMNS).iter_rows(named=True):
                f.write(json.dumps(row, default=str) + "\n")

    print(f"With prompt: {len(wp):,} clips, Without prompt: {len(np_):,} clips")


if __name__ == "__main__":
    main()
