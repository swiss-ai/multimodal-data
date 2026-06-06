#!/bin/bash
# SRG Apertus — convert → tokenize → materialize end-to-end (internal-only).
# Source dataset: i4ds/SRG_apertus_data, ~3552 hrs Swiss German news/radio.
#
# Single-process invocation of `stage=all`; resume-safe across restarts via
# the dual-marker (_SUCCESS + state-fingerprint) protocol on each stage.

#SBATCH --account=infra01
#SBATCH --job-name=srg-apertus-e2e
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/srg-apertus-e2e-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/srg-apertus-e2e-%A.err
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

uv pip install --python /opt/venv/bin/python polars wandb

ulimit -n 65536

SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/internal-only/srg_apertus
TOKENIZED_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/internal-only
INTERLEAVED_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/internal-only/srg_apertus_interleaved

mkdir -p "${SHAR_DIR}" "${TOKENIZED_DIR}" "${INTERLEAVED_DIR}"

# wandb: WANDB_API_KEY must be present (cluster secret or ~/.netrc).
export WANDB_PROJECT=${WANDB_PROJECT:-apertus-audio-tokenization}

echo "=========================================="
echo "SRG Apertus end-to-end (internal-only)"
echo "  Job:         ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo "  Repo:        ${REPO_DIR}"
echo "  SHAR:        ${SHAR_DIR}"
echo "  Tokenized:   ${TOKENIZED_DIR}/srg_apertus"
echo "  Interleaved: ${INTERLEAVED_DIR}"
echo "=========================================="

cd "${REPO_DIR}"
python -m audio_tokenization run \
    dataset=srg_apertus \
    stage=all \
    "dataset.convert.output.shar_dir=${SHAR_DIR}" \
    "dataset.tokenize.output.output_dir=${TOKENIZED_DIR}" \
    "dataset.materialize.interleave.output_dir=${INTERLEAVED_DIR}" \
    "+dataset.tokenize.wandb.enabled=true" \
    "+dataset.tokenize.wandb.project=${WANDB_PROJECT}" \
    "+dataset.tokenize.wandb.name=srg_apertus" \
    "+dataset.tokenize.wandb.tags=[srg_apertus,gsw,interleaved,internal-only]"

echo ""; echo "[$(date '+%F %T')] SRG Apertus end-to-end complete."
