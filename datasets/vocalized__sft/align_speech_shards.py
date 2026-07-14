"""
Usage (SLURM array):
  python align_speech_shards.py --shard-id $SLURM_ARRAY_TASK_ID \
      --num-shards 64 --out-root /path/to/output
Single-turn copy pass (run once, no GPU, no sharding needed):
  python align_speech_shards.py --copy-single-turns --out-root /path/to/output
"""
import os
import json
import shutil
import logging
import argparse
import unicodedata

import torch
import soundfile as sf
from qwen_asr import Qwen3ForcedAligner

parser = argparse.ArgumentParser()
parser.add_argument("--shard-id", type=int, default=0)
parser.add_argument("--num-shards", type=int, default=1)
parser.add_argument("--input", default="sft_datasets_rephrased_id.jsonl")
parser.add_argument("--audio-root", default="vocalized_sft")
parser.add_argument("--out-root", required=True)
parser.add_argument("--copy-single-turns", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, filename=f"align_shard{args.shard_id}.log",
                    filemode="a", format="%(asctime)s %(levelname)s %(message)s")

SHARD_SIZE = 1000
BATCH_SIZE = 32
CJK_LANGS = {"Chinese", "Japanese", "Korean"}
PUNCT = ".,!?。、！？"

LANG_NORMALIZE = {
    "en": "English", "english": "English", "ja": "Japanese", "japanese": "Japanese",
    "zh": "Chinese", "chinese": "Chinese", "de": "German", "german": "German",
    "fr": "French", "french": "French", "es": "Spanish", "spanish": "Spanish",
    "it": "Italian", "italian": "Italian", "ko": "Korean", "korean": "Korean",
    "pt": "Portuguese", "portuguese": "Portuguese", "ru": "Russian", "russian": "Russian",
}


def out_dir_for(idx):
    return os.path.join(args.out_root, f"shard_{idx // SHARD_SIZE:06d}")


def _norm(s):
    s = s.lower()
    while s and unicodedata.category(s[0]).startswith("P"):
        s = s[1:]
    while s and unicodedata.category(s[-1]).startswith("P"):
        s = s[:-1]
    return s


def find_boundaries_word(turns, items, n=3):
    b, ts_idx = [], 0
    for turn in turns:
        target = [_norm(w) for w in turn.split()[:n]]
        found = None
        for i in range(ts_idx, len(items)):
            window = [_norm(t.text) for t in items[i:i + n]]
            if window == target:
                found = items[i].start_time
                ts_idx = i + len(turn.split())
                break
        b.append(found)
    return b


def find_boundaries_cjk(turns, items, n=6):
    chars, times = [], []
    for it in items:
        for c in it.text.strip(PUNCT):
            chars.append(c)
            times.append(it.start_time)
    stream = "".join(chars)

    b, pos = [], 0
    for turn in turns:
        clean = "".join(c for c in turn if not c.isspace() and c not in PUNCT)
        target = clean[:n]
        idx = stream.find(target, pos)
        if idx >= 0:
            b.append(times[idx])
            pos = idx + max(len(clean) - 2, 1)
        else:
            b.append(None)
    return b


def segment_audio(wav_path, boundaries, out_dir, eid):
    audio, sr = sf.read(wav_path)
    total = len(audio) / sr
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else total
        sf.write(os.path.join(out_dir, f"{eid}_turn_{i}.wav"),
                 audio[int(start * sr):int(end * sr)], sr)


def load_shard():
    items = []
    with open(args.input) as f:
        for idx, line in enumerate(f):
            if idx % args.num_shards != args.shard_id:
                continue
            ex = json.loads(line)
            eid = ex["id"]
            if os.path.exists(os.path.join(out_dir_for(idx), f"{eid}.done")):
                continue
            lang = LANG_NORMALIZE.get(str(ex["language"]).strip().lower())
            if lang is None:
                continue
            turns = [t["value"] for t in ex["conversation"] if t["from"] == "human"]
            if len(turns) < 2:
                continue
            wav = os.path.join(args.audio_root, f"{eid}.wav")
            if not os.path.exists(wav):
                logging.warning(f"Missing audio for {eid}")
                continue
            items.append({"idx": idx, "id": eid, "lang": lang, "turns": turns, "wav": wav})
    return items


def copy_single_turns():
    n = 0
    with open(args.input) as f:
        for idx, line in enumerate(f):
            ex = json.loads(line)
            turns = [t for t in ex["conversation"] if t["from"] == "human"]
            if len(turns) != 1:
                continue
            src = os.path.join(args.audio_root, f"{ex['id']}.wav")
            if not os.path.exists(src):
                continue
            out_dir = out_dir_for(idx)
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, f"{ex['id']}_turn_0.wav")
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                n += 1
    print(f"Copied {n} single-turn files.")


def main():
    if args.copy_single_turns:
        copy_single_turns()
        return

    aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )

    items = load_shard()
    logging.info(f"Shard {args.shard_id}/{args.num_shards}: {len(items)} items")

    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start:start + BATCH_SIZE]
        texts = [". ".join(t.rstrip(".") for t in it["turns"]) + "." for it in batch]
        try:
            results = aligner.align(
                audio=[it["wav"] for it in batch],
                text=texts,
                language=[it["lang"] for it in batch],
            )
        except Exception as e:
            logging.error(f"Aligner batch {start} failed: {e}")
            continue

        for it, ts in zip(batch, results):
            try:
                fn = find_boundaries_cjk if it["lang"] in CJK_LANGS else find_boundaries_word
                bounds = fn(it["turns"], ts)
                if any(b is None for b in bounds):
                    logging.warning(f"Match failed {it['id']} "
                                    f"({sum(b is not None for b in bounds)}/{len(bounds)})")
                    continue
                out_dir = out_dir_for(it["idx"])
                os.makedirs(out_dir, exist_ok=True)
                segment_audio(it["wav"], bounds, out_dir, it["id"])
                open(os.path.join(out_dir, f"{it['id']}.done"), "w").close()
            except Exception as e:
                logging.error(f"Segmentation failed {it['id']}: {e}")

        logging.info(f"Processed {min(start + BATCH_SIZE, len(items))}/{len(items)}")

    print(f"Shard {args.shard_id} done.")


if __name__ == "__main__":
    main()