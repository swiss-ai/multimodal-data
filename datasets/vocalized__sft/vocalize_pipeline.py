import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from huggingface_hub import snapshot_download
import os
import json
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shard-id", type=int, required=True)
parser.add_argument("--num-shards", type=int, required=True)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, filename=f"vocalize_shard{args.shard_id}.log", filemode="a",
                    format="%(asctime)s %(levelname)s %(message)s")

SUPPORTED_LANGS = {"Chinese", "English", "Japanese", "Korean", "German",
                   "French", "Russian", "Portuguese", "Spanish", "Italian"}

# Setup model
model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
model_path = snapshot_download(repo_id=model_id)
tokenizer_path = os.path.join(model_path, "speech_tokenizer")
if not os.path.exists(os.path.join(tokenizer_path, "preprocessor_config.json")):
    os.makedirs(tokenizer_path, exist_ok=True)
    for config_file in ["preprocessor_config.json", "config.json"]:
        src = os.path.join(model_path, config_file)
        dst = os.path.join(tokenizer_path, config_file)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

# Load all data then take this shard's slice
all_items = []
os.makedirs("vocalized_sft", exist_ok=True)
with open("sft_datasets_rephrased_id.jsonl", "r") as f:
    for line in f:
        ex = json.loads(line)
        eid = ex["id"]
        if os.path.exists(f"vocalized_sft/{eid}.wav"):
            continue
        lang = ex["language"]
        if lang not in SUPPORTED_LANGS:
            logging.warning(f"Skipping {eid}: unsupported language '{lang}'")
            continue
        all_items.append(ex)

# Shard data
shard_items = all_items[args.shard_id::args.num_shards]
logging.info(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_items)} items")

ids, texts, langs = [], [], []
for ex in shard_items:
    parts = []
    for turn in ex["conversation"]:
        if turn["from"] == "human":
            parts.append(turn["value"].rstrip("."))
    ids.append(ex["id"])
    texts.append(". ".join(parts) + ".")
    langs.append(ex["language"])

BATCH_SIZE = 16

for start in range(0, len(texts), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(texts))
    batch_texts = texts[start:end]
    batch_langs = langs[start:end]
    batch_ids = ids[start:end]
    batch_instruct = ["Speak in a natural, conversational tone with appropriate emotion and pacing."] * len(batch_texts)

    try:
        wavs, sr = model.generate_voice_design(
            text=batch_texts,
            language=batch_langs,
            instruct=batch_instruct,
        )
        for j, wav in enumerate(wavs):
            sf.write(f"vocalized_sft/{batch_ids[j]}.wav", wav, sr)
        logging.info(f"Done {end}/{len(texts)}")
    except Exception as e:
        logging.error(f"Batch {start}-{end} failed: {e}")
        for j in range(len(batch_texts)):
            try:
                wavs, sr = model.generate_voice_design(
                    text=[batch_texts[j]],
                    language=[batch_langs[j]],
                    instruct=[batch_instruct[j]],
                )
                sf.write(f"vocalized_sft/{batch_ids[j]}.wav", wavs[0], sr)
            except Exception as e2:
                logging.error(f"Single {batch_ids[j]} failed: {e2}")

print(f"Shard {args.shard_id} done.")