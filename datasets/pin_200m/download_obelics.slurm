#!/bin/bash
# Download remaining m-a-p/PIN-200M OBELICS parts in parallel chunks
# Parts part00706–part00999 (294 parts), split into 5 array tasks
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-pin200m-obl
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pin200m-obl-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pin200m-obl-%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --array=0-4

set -euo pipefail

DATASET_REPO="m-a-p/PIN-200M"
REVISION="f69c5da58f4284c6687a0e058c21e67fca9a1b66"  # main as of 2025-12-15
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___m-a-p___PIN-200M}

# Each task downloads a range of ~60 parts
# Task 0: part00706–part00765
# Task 1: part00766–part00825
# Task 2: part00826–part00885
# Task 3: part00886–part00945
# Task 4: part00946–part00999
START=$((706 + SLURM_ARRAY_TASK_ID * 60))
if [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    END=999
else
    END=$((START + 59))
fi

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache_obelics_${SLURM_ARRAY_TASK_ID}"

# Auto-resubmit
MAX_RESUBMITS=${MAX_RESUBMITS:-10}
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}

if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMITS" ]; then
    NEXT_ID=$(sbatch --dependency=afterany:${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
        --export=ALL,RESUBMIT_COUNT=$((RESUBMIT_COUNT + 1)),MAX_RESUBMITS=${MAX_RESUBMITS} \
        --array=${SLURM_ARRAY_TASK_ID} \
        "$0" | awk '{print $4}')
    echo "Queued follow-up job ${NEXT_ID} (resubmit $((RESUBMIT_COUNT + 1))/${MAX_RESUBMITS})"
else
    echo "Reached max resubmit limit (${MAX_RESUBMITS}), no follow-up queued"
fi

echo "========================================"
echo "PIN-200M OBELICS chunk download (task ${SLURM_ARRAY_TASK_ID}, attempt $((RESUBMIT_COUNT + 1)))"
echo "Range:    part$(printf '%05d' $START)–part$(printf '%05d' $END)"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

# Download one part at a time to avoid --include glob issues
FAILED=0
for i in $(seq "$START" "$END"); do
    PART="part$(printf '%05d' $i)"

    # Skip if already fully downloaded
    if [ -d "${DEST_DIR}/data/OBELICS/${PART}" ] && [ -f "${DEST_DIR}/data/OBELICS/${PART}/${PART}.jsonl" ]; then
        echo "Skipping ${PART} (already exists)"
        continue
    fi

    echo "[$(date '+%F %T')] Downloading ${PART}..."
    huggingface-cli download "${DATASET_REPO}" \
      --repo-type dataset \
      --revision "${REVISION}" \
      --local-dir "${DEST_DIR}" \
      --include "data/OBELICS/${PART}/*" \
      --max-workers 64

    if [ $? -ne 0 ]; then
        echo "[$(date '+%F %T')] Failed on ${PART}, will resume on next attempt"
        FAILED=1
        break
    fi
    echo "[$(date '+%F %T')] Done: ${PART}"
done

if [ "$FAILED" -eq 0 ]; then
    echo "[$(date '+%F %T')] All parts downloaded for range part$(printf '%05d' $START)–part$(printf '%05d' $END)"
    echo "Pinned to commit: ${REVISION}"

    if [ -n "${NEXT_ID:-}" ]; then
        scancel "$NEXT_ID" 2>/dev/null && echo "Cancelled follow-up job ${NEXT_ID}"
    fi

    rm -rf "${HF_HUB_CACHE}"
    echo "Done."
else
    echo "[$(date '+%F %T')] Incomplete, follow-up job will resume"
fi
