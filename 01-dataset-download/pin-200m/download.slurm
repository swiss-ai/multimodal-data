#!/bin/bash
# Download m-a-p/PIN-200M from HuggingFace
# ~312 TB, ~200M interleaved text+image document samples
# Subsets: DocLayNet, OBELICS, PIN-Arxiv, PIN-PMC, chinese-markdown, leetcode, linux-cn, mmc4-core-ff, pg19
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-pin200m
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pin200m-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pin200m-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="m-a-p/PIN-200M"
REVISION="f69c5da58f4284c6687a0e058c21e67fca9a1b66"  # main as of 2025-12-15
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___m-a-p___PIN-200M}

# Install hf_transfer for faster downloads
uv pip install --python /opt/venv/bin/python hf_transfer

# Use a temporary cache dir to avoid polluting home and enable easy cleanup
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1

# Auto-resubmit: schedule next job to run after this one finishes (success or timeout).
# The next run resumes where this one left off. Max chain depth controlled by MAX_RESUBMITS.
MAX_RESUBMITS=${MAX_RESUBMITS:-20}
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
echo "PIN-200M Download (attempt $((RESUBMIT_COUNT + 1)))"
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

    # Cancel the queued follow-up job since we're done
    if [ -n "${NEXT_ID:-}" ]; then
        scancel "$NEXT_ID" 2>/dev/null && echo "Cancelled follow-up job ${NEXT_ID}"
    fi

    # Clean up HF hub cache (duplicate of downloaded files)
    echo "Cleaning up HF hub cache..."
    rm -rf "${HF_HUB_CACHE}"
    echo "Done."
else
    echo "[$(date '+%F %T')] Download incomplete (exit code ${DOWNLOAD_EXIT}), follow-up job will resume"
fi
