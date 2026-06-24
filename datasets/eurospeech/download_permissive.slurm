#!/bin/bash
# Download disco-eth/EuroSpeech — permissive commercial-use languages only (train split)
# Italy (CC BY 4.0), UK (OGL v3.0), Bulgaria (CC BY 2.5), France (Licence Ouverte), Norway (NLOD 2.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-eurospeech
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/eurospeech-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/eurospeech-%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --array=0-4

set -euo pipefail

DATASET_REPO="disco-eth/EuroSpeech"
REVISION="d244daad20b9189245f361c057df5b130d75c113"  # main as of 2026-03-28
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___disco-eth___EuroSpeech"

LANGUAGES=(
    italy
    uk
    bulgaria
    france
    norway
)

LANG="${LANGUAGES[$SLURM_ARRAY_TASK_ID]}"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache_${LANG}"

# Auto-resubmit
MAX_RESUBMITS=${MAX_RESUBMITS:-10}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]; then
    NEXT_ID=$(sbatch --dependency=afterany:${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
        --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),MAX_RESUBMITS=${MAX_RESUBMITS} \
        --array=${SLURM_ARRAY_TASK_ID} \
        "$0" | awk '{print $4}')
    echo "Queued follow-up job ${NEXT_ID} for ${LANG} (resubmit $((RESUBMIT_COUNT + 1))/${MAX_RESUBMITS})"
else
    echo "Reached max resubmit limit (${MAX_RESUBMITS}) for ${LANG}, no follow-up queued"
fi

echo "========================================"
echo "EuroSpeech download: ${LANG} (task ${SLURM_ARRAY_TASK_ID}, attempt $((RESUBMIT_COUNT + 1)))"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --include "${LANG}/train-*" \
  --max-workers 64

DOWNLOAD_EXIT=$?

if [ "$DOWNLOAD_EXIT" -eq 0 ]; then
    echo "[$(date '+%F %T')] Download finished for ${LANG}"
    echo "Pinned to commit: ${REVISION}"

    if [ -n "${NEXT_ID:-}" ]; then
        scancel "$NEXT_ID" 2>/dev/null && echo "Cancelled follow-up job ${NEXT_ID}"
    fi

    rm -rf "${HF_HUB_CACHE}"

    echo "Size:"
    du -sh "${DEST_DIR}/${LANG}" 2>/dev/null || echo "N/A"
    echo "Done."
else
    echo "[$(date '+%F %T')] Download incomplete for ${LANG} (exit code ${DOWNLOAD_EXIT}), follow-up job will resume"
fi
