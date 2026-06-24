#!/usr/bin/env python3
"""Patch existing AudioSet SHAR cuts with UTS captions and pre-tokenized text.

Reads the UTS manifest, builds a mapping from YouTube video_id to caption,
then walks through all cuts JSONL files in the AudioSet SHAR directories
and adds:
- ``supervisions[0].text`` — the UTS caption
- ``custom.text_tokens`` — pre-tokenized caption (via --text-tokenizer)
- ``supervisions[0].custom.uts_caption`` — same caption (for provenance)
- ``supervisions[0].custom.uts_audio_tag`` — UTS audio tags

Original files are backed up as *.jsonl.gz.bak before modification.

Usage:
    python patch_shar_with_uts.py \
        --uts-manifest /path/to/captionstew400k_full_tag_3k.jsonl.gz \
        --shar-dirs /path/to/audioset_bal_train /path/to/audioset_unbal_train \
        --text-tokenizer /path/to/tokenizer.json \
        --dry-run  # optional: just report stats without modifying
"""

import argparse
import gzip
import json
import glob
import shutil
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_text_tokenizer(tokenizer_path: str):
    """Load a Rust fast tokenizer from a tokenizer.json file."""
    if tokenizer_path is None:
        return None
    from tokenizers import Tokenizer
    path = Path(tokenizer_path)
    if not path.is_file():
        raise FileNotFoundError(f"Text tokenizer not found: {path}")
    tok = Tokenizer.from_file(str(path))
    logger.info(f"Text tokenizer loaded: {path}")
    return tok


def load_uts_captions(manifest_path: str) -> dict:
    """Load UTS manifest and build video_id -> caption mapping.

    UTS audioset IDs have a 'Y' prefix that needs to be stripped
    to match the YouTube video_id in the SHAR cuts.
    """
    caption_map = {}
    tag_map = {}
    with gzip.open(manifest_path, "rt") as f:
        for line in f:
            d = json.loads(line)
            src = d.get("recording", {}).get("sources", [{}])[0].get("source", "")
            if src != "audioset":
                continue
            uid = d["id"]
            if uid.startswith("Y"):
                video_id = uid[1:]
            else:
                video_id = uid
            sup = d.get("supervisions", [{}])[0]
            custom = sup.get("custom", {})
            caption = custom.get("caption", "")
            tags = custom.get("audio_tag", [])
            if caption:
                caption_map[video_id] = caption
            if tags:
                tag_map[video_id] = tags
    logger.info(
        f"Loaded {len(caption_map)} audioset captions, "
        f"{len(tag_map)} tag entries from UTS manifest"
    )
    return caption_map, tag_map


def patch_cuts_file(
    cuts_path: str,
    caption_map: dict,
    tag_map: dict,
    text_tokenizer=None,
    dry_run: bool = False,
) -> tuple:
    """Patch a single cuts JSONL file with UTS captions.

    Returns (total, patched, already_has_text) counts.
    """
    total = 0
    patched = 0
    already_has_text = 0
    lines_out = []

    with gzip.open(cuts_path, "rt") as f:
        for line in f:
            total += 1
            d = json.loads(line)
            video_id = d.get("custom", {}).get("video_id", "")

            if video_id in caption_map:
                if d.get("supervisions"):
                    sup = d["supervisions"][0]
                    caption = caption_map[video_id]

                    if sup.get("text"):
                        already_has_text += 1
                    else:
                        sup["text"] = caption
                        patched += 1

                    # Add tags and caption to supervision custom
                    if "custom" not in sup:
                        sup["custom"] = {}
                    sup["custom"]["uts_caption"] = caption
                    if video_id in tag_map:
                        sup["custom"]["uts_audio_tag"] = tag_map[video_id]

                    # Pre-tokenize text into cut custom
                    if text_tokenizer is not None and sup.get("text"):
                        if "custom" not in d or d["custom"] is None:
                            d["custom"] = {}
                        d["custom"]["text_tokens"] = text_tokenizer.encode(
                            sup["text"], add_special_tokens=False
                        ).ids

            lines_out.append(json.dumps(d, ensure_ascii=False) + "\n")

    if not dry_run and patched > 0:
        backup_path = cuts_path + ".bak"
        if not Path(backup_path).exists():
            shutil.copy2(cuts_path, backup_path)

        with gzip.open(cuts_path, "wt") as f:
            f.writelines(lines_out)

    return total, patched, already_has_text


def main():
    parser = argparse.ArgumentParser(description="Patch AudioSet SHAR with UTS captions")
    parser.add_argument(
        "--uts-manifest",
        required=True,
        help="Path to UTS manifest (captionstew400k_full_tag_3k.jsonl.gz)",
    )
    parser.add_argument(
        "--shar-dirs",
        nargs="+",
        required=True,
        help="SHAR directories to patch (e.g. audioset_bal_train audioset_unbal_train)",
    )
    parser.add_argument(
        "--text-tokenizer",
        default=None,
        help="Path to tokenizer.json for pre-tokenizing text into custom.text_tokens",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report stats without modifying files",
    )
    args = parser.parse_args()

    caption_map, tag_map = load_uts_captions(args.uts_manifest)
    text_tokenizer = load_text_tokenizer(args.text_tokenizer)

    total_all = 0
    patched_all = 0
    already_all = 0

    for shar_dir in args.shar_dirs:
        cuts_files = sorted(glob.glob(f"{shar_dir}/**/cuts.*.jsonl.gz", recursive=True))
        logger.info(f"Processing {len(cuts_files)} cuts files in {shar_dir}")

        for cuts_path in cuts_files:
            total, patched, already = patch_cuts_file(
                cuts_path, caption_map, tag_map,
                text_tokenizer=text_tokenizer,
                dry_run=args.dry_run,
            )
            total_all += total
            patched_all += patched
            already_all += already

            if patched > 0:
                logger.info(
                    f"  {cuts_path}: {patched}/{total} patched"
                    f"{' (dry-run)' if args.dry_run else ''}"
                )

    logger.info("=" * 60)
    logger.info(f"Total cuts: {total_all}")
    logger.info(f"Patched with UTS caption: {patched_all}")
    logger.info(f"Already had text: {already_all}")
    logger.info(f"Unmatched: {total_all - patched_all - already_all}")
    if args.dry_run:
        logger.info("DRY RUN — no files modified")


if __name__ == "__main__":
    main()
