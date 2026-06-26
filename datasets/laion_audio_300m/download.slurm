#!/bin/bash
# Download laion/LAION-Audio-300M (~29TB, 2775 tar files, Apache 2.0)
# YouTube-scraped audio snippets (mp3 + metadata.json) with Gemini-Flash-2.0 captions.
# Split across 16 array tasks (~174 tars ≈ 1.8 TB per task).
#
# Usage:
#   sbatch --array=0-15 download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-laion-audio-300m
#SBATCH --reservation=SD-69241-apertus-1-5-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/laion-audio-300m-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/laion-audio-300m-%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

DATASET_REPO="laion/LAION-Audio-300M"
REVISION="29eaacba2d0815aaf608ab34303555b9c895792e"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___laion___LAION-Audio-300M"
TOTAL_TARS=2775
NUM_TASKS=16

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache_${SLURM_ARRAY_TASK_ID}"

mkdir -p "${DEST_DIR}"

# Compute index range for this array task
PER_TASK=$(( (TOTAL_TARS + NUM_TASKS - 1) / NUM_TASKS ))
START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
END=$(( START + PER_TASK ))
if [ $END -gt $TOTAL_TARS ]; then END=$TOTAL_TARS; fi

echo "========================================"
echo "LAION-Audio-300M download (task ${SLURM_ARRAY_TASK_ID}/${NUM_TASKS})"
echo "Repo: ${DATASET_REPO}"
echo "Dest: ${DEST_DIR}"
echo "Range: tars [${START}, ${END}) = $(( END - START )) files"
echo "========================================"

# Build --include args for this task's tar range
INCLUDE_ARGS=()
for (( j=START; j<END; j++ )); do
    INCLUDE_ARGS+=( --include "flash_15_2_random_snippets_${j}.tar" )
done

# Task 0 also fetches the README and .gitattributes
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    INCLUDE_ARGS+=( --include "README.md" --include ".gitattributes" )
fi

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    "${INCLUDE_ARGS[@]}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 64

rm -rf "${HF_HUB_CACHE}"
echo "[$(date '+%F %T')] Done: task ${SLURM_ARRAY_TASK_ID}"
