#!/bin/bash
# Download allenai/Molmo2-MultiImageQA
#SBATCH --account=infra01
#SBATCH --job-name=dl-molmo2_multi
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/molmo2_multiimage_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/molmo2_multiimage_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="allenai/Molmo2-MultiImageQA"
REVISION="f47ca3644d39"
DEST="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___allenai___Molmo2-MultiImageQA"
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
echo "Done: molmo2_multiimage"
