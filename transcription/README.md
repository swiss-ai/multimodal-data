# Audio Transcription Pipeline

Batch-transcribe audio datasets using faster-whisper using multi-node, multi-GPU parallelism.

## Structure

```
job.sh              SLURM
transcribe.py       Transcription logic
loaders/
  __init__.py       Format registry
  arrow.py          Arrow file loader
```

## Quick Start

1. Edit the config at the top of `job.sh` (paths, format, language)
2. `sbatch job.sh`
3. To resume after a failure, just resubmit — completed shards are skipped

## Adding Input Formats

Create a loader in `loaders/` that returns `(file_list, load_fn)` where `load_fn(path)` returns a HuggingFace Dataset with an `audio` column. Register it in `loaders/__init__.py`.

## Environment

Use ctranslate2-nemo-cudnn.toml as the environment for access to faster_whisper model