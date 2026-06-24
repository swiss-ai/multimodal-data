# Audio Transcription Pipeline

Batch-transcribe audio datasets using faster-whisper using multi-node, multi-GPU parallelism.

## Structure

```
job.slurm           SLURM array job (one node per task, one worker per GPU)
transcribe.py       Transcription logic
loaders/
  __init__.py       Format registry
  arrow.py          Arrow file loader
```

## Quick Start

1. Set `INPUT_DIR` and `OUTPUT_DIR`, then submit:
   ```bash
   INPUT_DIR=/path/to/shards OUTPUT_DIR=/path/to/out sbatch job.slurm
   ```
2. To resume after a failure, just resubmit — completed shards are skipped.

## Parameters

| Variable | Default | Description |
|---|---|---|
| `INPUT_DIR` | *(required)* | Directory containing input shards |
| `OUTPUT_DIR` | *(required)* | Directory for transcribed output |
| `INPUT_FORMAT` | `arrow` | Loader format (see `loaders/`) |
| `LANGUAGE` | `en` | Whisper language code |
| `WHISPER_MODEL` | `turbo` | faster-whisper model name |
| `BATCH_SIZE` | `16` | Inference batch size per worker |
| `GPUS_PER_NODE` | `4` | Workers per node (one per GPU) |
| `CLUSTER_REPO_HOME` | `$SLURM_SUBMIT_DIR` | Path to this repo on the cluster |

## Adding Input Formats

Create a loader in `loaders/` that returns `(file_list, load_fn)` where `load_fn(path)` returns a HuggingFace Dataset with an `audio` column. Register it in `loaders/__init__.py`.

## Environment

Use `ctranslate2-nemo-cudnn` as the SLURM environment (`--environment=ctranslate2-nemo-cudnn`) for access to faster-whisper and CUDA libraries.
