#!/bin/bash
# Download MERaLiON/Multitask-National-Speech-Corpus-v1 from Hugging Face
# ~3 TB, Singlish/Singapore multilingual speech dataset
# 36 configs: ASR (PART1-6), SQA (PART3-6), SDS (PART3-6), PQA variants
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-mnsc
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mnsc-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mnsc-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="MERaLiON/Multitask-National-Speech-Corpus-v1"
REVISION="d169debbce4a2dc07f7293360d3eed33c06793b9"  # main as of 2026-03-22
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___MERaLiON___Multitask-National-Speech-Corpus-v1"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

# Auto-resubmit
MAX_RESUBMITS=${MAX_RESUBMITS:-10}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]; then
    NEXT_ID=$(sbatch --dependency=afterany:${SLURM_JOB_ID} \
        --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),MAX_RESUBMITS=${MAX_RESUBMITS} \
        "$0" | awk '{print $4}')
    echo "Queued follow-up job ${NEXT_ID} (resubmit $((RESUBMIT_COUNT + 1))/${MAX_RESUBMITS})"
else
    echo "Reached max resubmit limit (${MAX_RESUBMITS}), no follow-up queued"
fi

echo "========================================"
echo "MNSC Download (attempt $((RESUBMIT_COUNT + 1)))"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --max-workers 64

DOWNLOAD_EXIT=$?

if [ "$DOWNLOAD_EXIT" -eq 0 ]; then
    echo "[$(date '+%F %T')] Download finished successfully"
    echo "Pinned to commit: ${REVISION}"

    if [ -n "${NEXT_ID:-}" ]; then
        scancel "$NEXT_ID" 2>/dev/null && echo "Cancelled follow-up job ${NEXT_ID}"
    fi

    rm -rf "${HF_HUB_CACHE}"

    echo "Final size:"
    du -sh "${DEST_DIR}"
    echo "Done."
else
    echo "[$(date '+%F %T')] Download incomplete (exit code ${DOWNLOAD_EXIT}), follow-up job will resume"
fi
