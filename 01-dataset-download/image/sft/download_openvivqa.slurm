#!/bin/bash
# Download uitnlp/OpenViVQA-dataset (~1.7GB, 9 files)
#SBATCH --account=infra01
#SBATCH --job-name=dl-openvivqa
#SBATCH --reservation=SD-69241-apertus-1-5-3
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/openvivqa_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/openvivqa_%j.err
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

DATASET_REPO="uitnlp/OpenViVQA-dataset"
REVISION="cc85c24c863a"
DEST="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___uitnlp___OpenViVQA-dataset"
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
echo "Done: openvivqa"
