# Vocalized SFT Pipeline

Convert text SFT conversations into per-turn speech audio.

1. `vocalize_pipeline.py` — reads the SFT JSONL and generates one WAV per
   conversation: all **human turns** concatenated, spoken in a single
   consistent voice (Qwen3-TTS VoiceDesign).
2. `align_speech_shards.py` — force aligns each WAV against its text
   (Qwen3-ForcedAligner), finds turn boundaries, and splits the audio into
   per-turn files.

## Requirements

Models (auto-downloaded from HuggingFace on first run; ~4 GB + ~1.5 GB):
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-ForcedAligner-0.6B`

Manual download:
```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B
```

Python: `pip install qwen-tts qwen-asr` (pulls in torch, transformers,
soundfile, etc.). One NVIDIA GPU per shard; no flash-attn needed (sdpa).

## Input

JSONL, one conversation per line:
```json
{"id": "123", "language": "English",
 "conversation": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
```
Supported languages: Chinese, English, Japanese, Korean, German, French,
Russian, Portuguese, Spanish, Italian. Others are skipped.

## Usage

```bash
# 1. Vocalize (SLURM array, e.g. 32 shards; resumable, skips existing WAVs)
python vocalize_pipeline.py --shard-id $SLURM_ARRAY_TASK_ID --num-shards 32
# reads ./sft_datasets_rephrased_id.jsonl, writes ./vocalized_sft/{id}.wav

# 2. Segment multi-turn conversations (SLURM array, e.g. 64 shards; resumable)
python align_speech_shards.py --shard-id $SLURM_ARRAY_TASK_ID --num-shards 64 --out-root /path/to/output

# 3. Copy single-turn conversations to output (run once, no GPU)
# Necessary since Step 2 only accounts for conversations with 2+ turns
python align_speech_shards.py --copy-single-turns --out-root /path/to/output
```

Output: `{out_root}/shard_XXXXXX/{id}_turn_{n}.wav`, sharded by JSONL line
index (1000 samples per dir). `.done` markers track completed samples;
failed boundary matches are logged and retried on rerun.