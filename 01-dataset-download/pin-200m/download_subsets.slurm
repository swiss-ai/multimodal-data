#!/bin/bash
# Download remaining m-a-p/PIN-200M subsets in parallel (one per array task)
# Subsets: PIN-Arxiv, PIN-PMC, chinese-markdown, leetcode, linux-cn, mmc4-core-ff, pg19
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-pin200m-sub
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pin200m-sub-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pin200m-sub-%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --array=0-6

set -euo pipefail

DATASET_REPO="m-a-p/PIN-200M"
REVISION="f69c5da58f4284c6687a0e058c21e67fca9a1b66"  # main as of 2025-12-15
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___m-a-p___PIN-200M}

SUBSETS=(
    PIN-Arxiv
    PIN-PMC
    chinese-markdown
    leetcode
    linux-cn
    mmc4-core-ff
    pg19
)

SUBSET="${SUBSETS[$SLURM_ARRAY_TASK_ID]}"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache_${SUBSET}"

# Auto-resubmit: schedule next job for this array task after it finishes.
# The next run resumes where this one left off.
MAX_RESUBMITS=${MAX_RESUBMITS:-10}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]; then
    NEXT_ID=$(sbatch --dependency=afterany:${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
        --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),MAX_RESUBMITS=${MAX_RESUBMITS} \
        --array=${SLURM_ARRAY_TASK_ID} \
        "$0" | awk '{print $4}')
    echo "Queued follow-up job ${NEXT_ID} for ${SUBSET} (resubmit $((RESUBMIT_COUNT + 1))/${MAX_RESUBMITS})"
else
    echo "Reached max resubmit limit (${MAX_RESUBMITS}) for ${SUBSET}, no follow-up queued"
fi

echo "========================================"
echo "PIN-200M subset download: ${SUBSET} (task ${SLURM_ARRAY_TASK_ID}, attempt $((RESUBMIT_COUNT + 1)))"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --revision "${REVISION}" \
  --local-dir "${DEST_DIR}" \
  --include "data/${SUBSET}/*" \
  --max-workers 64

DOWNLOAD_EXIT=$?

if [ "$DOWNLOAD_EXIT" -eq 0 ]; then
    echo "[$(date '+%F %T')] Download finished for subset: ${SUBSET}"
    echo "Pinned to commit: ${REVISION}"

    # Cancel the queued follow-up job since we're done
    if [ -n "${NEXT_ID:-}" ]; then
        scancel "$NEXT_ID" 2>/dev/null && echo "Cancelled follow-up job ${NEXT_ID}"
    fi

    rm -rf "${HF_HUB_CACHE}"

    echo "Final size of subset:"
    du -sh "${DEST_DIR}/data/${SUBSET}" 2>/dev/null || echo "N/A"
    echo "Done."
else
    echo "[$(date '+%F %T')] Download incomplete for ${SUBSET} (exit code ${DOWNLOAD_EXIT}), follow-up job will resume"
fi
