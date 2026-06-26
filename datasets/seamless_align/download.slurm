#!/bin/bash
# Download ai4bharat/SeamlessAlign (~808GB, 5 Indian languages → English speech translation, CC-BY-4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-seamless
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/seamless-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/seamless-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="ai4bharat/SeamlessAlign"
REVISION="8c812b0c895ffc3e1d8f63f3c0122d9cf332a286"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___ai4bharat___SeamlessAlign"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

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
echo "SeamlessAlign Download (attempt $((RESUBMIT_COUNT + 1)))"
echo "Repo: ${DATASET_REPO}, Dest: ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

set +e
huggingface-cli download "${DATASET_REPO}" --repo-type dataset --revision "${REVISION}" --local-dir "${DEST_DIR}" --max-workers 64

DOWNLOAD_EXIT=$?
set -e

if [ "$DOWNLOAD_EXIT" -eq 0 ]; then
    echo "[$(date '+%F %T')] Download finished"
    [ -n "${NEXT_ID:-}" ] && scancel "$NEXT_ID" 2>/dev/null && echo "Cancelled follow-up ${NEXT_ID}"
    rm -rf "${HF_HUB_CACHE}"
    du -sh "${DEST_DIR}"
else
    echo "[$(date '+%F %T')] Incomplete, follow-up will resume"
fi
