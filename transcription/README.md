# Audio Transcription Pipeline

Batch-transcribe audio datasets with [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
across multiple nodes and GPUs. Each input shard keeps all of its original
columns and gains a `text` column with the transcript.

## Structure

```
job.slurm           SLURM array job (one node per array task, one worker per GPU)
transcribe.py       Loader + faster-whisper inference + sharded output writer
loaders/
  __init__.py       Format registry (LOADERS)
  arrow.py          Loader for HuggingFace .arrow shards
```

## Quick start

Set `INPUT_DIR` and `OUTPUT_DIR`, then submit:

```bash
INPUT_DIR=/path/to/shards OUTPUT_DIR=/path/to/out sbatch job.slurm
```

Override the model or language as needed:

```bash
LANGUAGE=de WHISPER_MODEL=large-v3 INPUT_DIR=/data/audio OUTPUT_DIR=/data/out sbatch job.slurm
```

To resume after a failure, resubmit the same command. Completed shards are
skipped (see [Resume](#resume)).

## How work is split

- Each array task (`--array=0-7` by default) runs on one node and launches
  `GPUS_PER_NODE` workers, one per GPU.
- The total worker count is `SLURM_ARRAY_TASK_COUNT * GPUS_PER_NODE`. Each
  worker gets a unique global id and processes the shards whose position in the
  sorted file list satisfies `index modulo total_workers == worker_id`.
- Workers write independently, so a failed node can be rerun without touching
  the others' output.

## Parameters

| Variable | Default | Description |
|---|---|---|
| `INPUT_DIR` | *(required)* | Directory containing input shards |
| `OUTPUT_DIR` | *(required)* | Directory for transcribed output |
| `INPUT_FORMAT` | `arrow` | Loader format, one of the keys in `loaders/__init__.py` |
| `LANGUAGE` | `en` | Whisper language code |
| `WHISPER_MODEL` | `turbo` | faster-whisper model name (e.g. `turbo`, `large-v3`) |
| `BATCH_SIZE` | `16` | Inference batch size per worker |
| `GPUS_PER_NODE` | `4` | Workers per node, one per GPU |
| `CLUSTER_REPO_HOME` | `$SLURM_SUBMIT_DIR` | Path to this repo on the cluster |

## Input and output

**Input.** The `arrow` loader reads `*.arrow` files from `INPUT_DIR` and casts
their `audio` column to 16 kHz. Any input format works as long as its loader
returns a HuggingFace `Dataset` with an `audio` column.

**Output.** For an input shard `foo.arrow`, the pipeline writes a
`save_to_disk` dataset directory `OUTPUT_DIR/foo_processed/`. It contains every
original column plus a `text` column. Writes are atomic: output goes to a
temporary directory in `OUTPUT_DIR` and is renamed into place only on success,
so a partial `foo_processed/` never appears.

Load the result with:

```python
from datasets import load_from_disk
ds = load_from_disk("/path/to/out/foo_processed")
print(ds[0]["text"])
```

## Resume

A shard is skipped when its `<stem>_processed/` output directory already exists.
Because output is renamed into place atomically, only fully written shards are treated
as done, so resubmitting after a crash reprocesses just the incomplete shards.

## Failure handling

- A sample that fails to transcribe gets an empty `text` string, and the shard
  continues.
- A shard that fails fatally is logged and skipped. The node keeps processing
  its remaining shards, and a node reports a non-zero exit if any of its
  workers failed.

## Adding input formats

Create a loader in `loaders/` that returns `(file_list, load_fn)` where
`load_fn(path)` returns a HuggingFace `Dataset` with an `audio` column.
Register it in `loaders/__init__.py` under the `LOADERS` dict, then pass its key
as `INPUT_FORMAT`.

## Environment

Submit with the `ctranslate2-nemo-cudnn` container
(`--environment=ctranslate2-nemo-cudnn`, already set in `job.slurm`) for
faster-whisper and the required CUDA libraries. Each worker uses a per-worker
`HF_DATASETS_CACHE` to avoid lock contention.
