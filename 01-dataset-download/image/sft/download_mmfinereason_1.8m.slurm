#!/bin/bash
# Download OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking (~83GB, 1.8M samples, Apache 2.0)
# Superset of 586K and 123K versions. Multimodal CoT reasoning distilled from Qwen3-VL-235B.
#SBATCH --account=infra01
#SBATCH --job-name=dl-mmfinereason-1.8m
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/mmfinereason_1.8m_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/mmfinereason_1.8m_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking"
REVISION="7c93f724791d01246f5425e1182b14ec8d013847"
DEST="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___OpenDataArena___MMFineReason-1.8M-Qwen3-VL-235B-Thinking"
export HF_HUB_CACHE="${DEST}/.hf_cache"

echo "=========================================="
echo "Downloading: ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest: ${DEST}"
echo "=========================================="

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST}" \
    --max-workers 64

rm -rf "${HF_HUB_CACHE}"
echo "Done: mmfinereason-1.8m"
