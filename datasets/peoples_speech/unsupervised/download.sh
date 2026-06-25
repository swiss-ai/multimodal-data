#!/bin/bash
# Download MLCommons/unsupervised_peoples_speech (multi-TB: audio/ + audio2/ tar
# shards, ~11k tars, plus lang_id/vad/licenses jsonl) for audio tokenization.
# Multi-TB → auto-resubmits with afterany dependency until the download completes.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/peoples_speech/unsupervised/download.sh
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-peoples-speech
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/peoples-speech-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/peoples-speech-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail

export HF_TOKEN="$(cat "$HOME/.hf-token")"
export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="MLCommons/unsupervised_peoples_speech"
REVISION="d917e17e86f6abc4fa4d83e958c8f4173f45f0e7"  # main as of 2026-06-06
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___MLCommons___unsupervised_peoples_speech"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"
mkdir -p "${DEST_DIR}"

# Auto-resubmit (multi-TB; one 12h window is not enough)
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
echo "Unsupervised People's Speech Download (attempt $((RESUBMIT_COUNT + 1)))"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

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
