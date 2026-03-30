#!/bin/bash
# Download speechcolab/gigaspeech2 from Hugging Face (gated, auto-approval)
# ~30K hours audio, Thai/Indonesian/Vietnamese, Apache 2.0 (non-commercial research only)
# NOTE: Requires HF token with accepted gated access
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-gigaspch2
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/gigaspeech2-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/gigaspeech2-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="speechcolab/gigaspeech2"
REVISION="aeedc8cde7fd37c4164ab1d1b5551dd12ee931fe"  # main as of 2026-03-22
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___speechcolab___gigaspeech2"

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
echo "GigaSpeech2 Download (attempt $((RESUBMIT_COUNT + 1)))"
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
