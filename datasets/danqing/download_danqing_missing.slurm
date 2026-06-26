#!/bin/bash
# Download missing DanQing shards by passing explicit file names to modelscope download.
# The ModelScope SDK has a pagination bug (drops cookies on page 2+ of list_repo_tree),
# but downloading explicit files works fine — bypasses the broken listing entirely.
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-danqing-missing
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/danqing-missing-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/danqing-missing-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -uo pipefail

DATASET_REPO="deepglint/DanQing"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/ms___deepglint___DanQing"
TOTAL_SHARDS=33298

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export MODELSCOPE_CACHE="${DEST_DIR}/.ms_cache"

# Auto-resubmit: schedule next job after this one finishes
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
echo "DanQing missing shards download (attempt $((RESUBMIT_COUNT + 1)))"
echo "Repo:     ${DATASET_REPO}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

# Build list of missing shards
MISSING_FILES=()
for i in $(seq 0 $((TOTAL_SHARDS - 1))); do
    shard=$(printf "data/train-%05d-of-%05d.parquet" "$i" "$TOTAL_SHARDS")
    if [[ ! -f "${DEST_DIR}/${shard}" ]]; then
        MISSING_FILES+=("${shard}")
    fi
done

echo "Missing shards: ${#MISSING_FILES[@]} / ${TOTAL_SHARDS}"

if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
    echo "All shards already downloaded!"
    # Cancel follow-up job
    if [ -n "${NEXT_ID:-}" ]; then
        scancel "$NEXT_ID" 2>/dev/null && echo "Cancelled follow-up job ${NEXT_ID}"
    fi
    rm -rf "${MODELSCOPE_CACHE}"
    du -sh "${DEST_DIR}"
    exit 0
fi

echo "First missing: ${MISSING_FILES[0]}"
echo "Last missing:  ${MISSING_FILES[-1]}"

# Download in batches of 500 to avoid command-line length limits
# Use set +e so a failed batch doesn't kill the whole job
BATCH_SIZE=500
FAILED_BATCHES=0
for ((start=0; start<${#MISSING_FILES[@]}; start+=BATCH_SIZE)); do
    batch=("${MISSING_FILES[@]:start:BATCH_SIZE}")
    echo "[$(date '+%F %T')] Downloading batch: shards $start to $((start + ${#batch[@]} - 1)) of ${#MISSING_FILES[@]}"

    modelscope download \
      --dataset "${DATASET_REPO}" \
      --local_dir "${DEST_DIR}" \
      --revision master \
      --max-workers 64 \
      "${batch[@]}" || { echo "Batch failed, continuing..."; FAILED_BATCHES=$((FAILED_BATCHES + 1)); }
done

echo "[$(date '+%F %T')] Download round finished (${FAILED_BATCHES} batch failures)"

rm -rf "${MODELSCOPE_CACHE}"

# Verify completeness
ACTUAL=$(ls "${DEST_DIR}/data/" | grep -c 'train-.*\.parquet' || true)
echo "Shards on disk: ${ACTUAL} / ${TOTAL_SHARDS}"

if [ "$ACTUAL" -ge "$TOTAL_SHARDS" ]; then
    echo "All shards downloaded!"
    if [ -n "${NEXT_ID:-}" ]; then
        scancel "$NEXT_ID" 2>/dev/null && echo "Cancelled follow-up job ${NEXT_ID}"
    fi
fi

du -sh "${DEST_DIR}"
echo "Done."
