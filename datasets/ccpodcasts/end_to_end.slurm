#!/bin/bash
# CCpodcasts — convert → tokenize → materialize end-to-end (internal-only).
# Source: 84 parquet shards in raw/ccpodcasts_parquet (produced by
# convert_to_parquet.py from segments.jsonl + dataset_b_segments WAVs).
#
# Single-process invocation of `stage=all`; resume-safe across restarts via
# the dual-marker (_SUCCESS + state-fingerprint) protocol on each stage.

#SBATCH --account=infra01
#SBATCH --job-name=ccpodcasts-e2e
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/ccpodcasts-e2e-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/ccpodcasts-e2e-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=1
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=${REPO_DIR:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}

# Single-process job (no DDP). Source the canonical runtime helper to set up
# PYTHONPATH/ffmpeg and install torchcodec + torchaudio into the container venv.
export INSTALL_TORCHCODEC=1
export INSTALL_TORCHAUDIO=1
source "${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"

uv pip install --python /opt/venv/bin/python polars wandb orjson

ulimit -n 65536

SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/internal-only/ccpodcasts
TOKENIZED_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/internal-only
INTERLEAVED_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/internal-only/ccpodcasts_interleaved

mkdir -p "${SHAR_DIR}" "${TOKENIZED_DIR}" "${INTERLEAVED_DIR}"

# wandb: WANDB_API_KEY must be present (cluster secret or ~/.netrc).
export WANDB_PROJECT=${WANDB_PROJECT:-apertus-audio-tokenization}

echo "=========================================="
echo "CCpodcasts end-to-end (internal-only)"
echo "  Job:         ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo "  Repo:        ${REPO_DIR}"
echo "  SHAR:        ${SHAR_DIR}"
echo "  Tokenized:   ${TOKENIZED_DIR}/ccpodcasts"
echo "  Interleaved: ${INTERLEAVED_DIR}"
echo "=========================================="

cd "${REPO_DIR}"

# 1/3 — convert: parquet → SHAR. Resume-safe; short-circuits if _SUCCESS.
python -m audio_tokenization run \
    dataset=ccpodcasts \
    stage=convert \
    "dataset.convert.output.shar_dir=${SHAR_DIR}"

# 2/3 — gap-aware re-grouping: rewrite cut.custom["source_id"]/clip_num so
# interleave runs break at parent_audio gaps > 5s (likely ad breaks /
# dropped chunks). Idempotent — group_by uses the immutable parent_audio
# custom field; touches only cut.custom, never cut.id. Tokenize reads the
# patched cut.custom values directly (audio_text.py:33-38).
# TODO: replace with materialize-time gap detection once that's wired
# (avoids re-tokenize on max_gap_sec changes).
python -m audio_tokenization.prepare.postprocess.patch_interleave_by_field \
    --shar-dir "${SHAR_DIR}" \
    --group-by parent_audio \
    --sort-by parent_start \
    --max-gap-sec 5.0 \
    --workers "${SLURM_CPUS_PER_TASK}"

# 3/3 — tokenize + materialize.
python -m audio_tokenization run \
    dataset=ccpodcasts \
    stage=tokenize \
    "dataset.tokenize.output.output_dir=${TOKENIZED_DIR}" \
    "+dataset.tokenize.wandb.enabled=true" \
    "+dataset.tokenize.wandb.project=${WANDB_PROJECT}" \
    "+dataset.tokenize.wandb.name=ccpodcasts" \
    "+dataset.tokenize.wandb.tags=[ccpodcasts,en,interleaved,internal-only,gap5s]"

python -m audio_tokenization run \
    dataset=ccpodcasts \
    stage=materialize \
    "dataset.materialize.interleave.output_dir=${INTERLEAVED_DIR}"

echo ""; echo "[$(date '+%F %T')] CCpodcasts end-to-end complete."
