#!/bin/bash
#SBATCH --job-name=waffle-caption
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/waffle/logs/waffle-caption-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/waffle/logs/waffle-caption-%j.err
#SBATCH --signal=B:TERM@120   # SIGTERM 120s before time limit, to flush buffer

# --environment=nemo_26_02 pins the container where polars, pyarrow, httpx,
# and Pillow live. Without it, the script relies on SPANK env-var inheritance
# from the submitter's shell, which is not reproducible.

# Production WAFFLE captioner sbatch.
#
# Requires the sml serving job to already be running. Pass its job-id so this
# client reads the router URL deterministically from its log.
#
# Usage:
#   sbatch caption_waffle.slurm <MODEL_JOB_ID>
#   # optional extra args forwarded to python:
#   sbatch caption_waffle.slurm <MODEL_JOB_ID> --concurrency 48 --max-rows 100
#
# Resume: safe to resubmit; already-captioned page_ids are skipped. If SLURM
# kills the job at the time limit, in-flight captions flush to a shard via
# the SIGTERM handler before exit.

set -euo pipefail

MODEL_JOB_ID="${1:-}"
if [[ -z "$MODEL_JOB_ID" ]]; then
  echo "ERROR: missing MODEL_JOB_ID" >&2
  echo "Usage: sbatch $0 <MODEL_JOB_ID> [--concurrency N] [--max-rows N]" >&2
  exit 2
fi
shift   # remaining args forwarded to python

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/waffle/logs

echo "caption_job=$SLURM_JOB_ID  model_job=$MODEL_JOB_ID  node=$(hostname)  started=$(date -Iseconds)"

# Use system Python: polars, pyarrow, httpx, Pillow are already installed there.
# The model-serving .venv (used by `sml advanced`) is minimal and would miss
# these deps. If a future captioner script grows a dep not in the system env,
# build a dedicated venv (e.g., /iopsstor/.../apertus/captioner/.venv) and
# source it here instead of the model-serving one.
PYTHON=$(command -v python3)
echo "python=$PYTHON"

SCRIPT=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/waffle/caption_waffle.py

"$PYTHON" "$SCRIPT" --job-id "$MODEL_JOB_ID" "$@" &
CAPTIONER_PID=$!

trap "echo 'forwarding SIGTERM to captioner'; kill -TERM $CAPTIONER_PID; wait $CAPTIONER_PID" TERM

wait $CAPTIONER_PID
echo "finished=$(date -Iseconds)"
