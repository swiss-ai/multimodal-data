#!/bin/bash
# Download nllg/DaTikZ-V4 (TikZ code-to-image dataset)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-datikz
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/download/datikz-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/download/datikz-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO="nllg/DaTikZ-V4"
REVISION="33734c83608211682be11001a1618856fc1979dd"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___nllg___DaTikZ-V4"

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

echo "========================================"
echo "DaTikZ-V4 Download"
echo "Repo: ${DATASET_REPO}, Dest: ${DEST_DIR}"
echo "========================================"

mkdir -p "${DEST_DIR}"

huggingface-cli download "${DATASET_REPO}" --repo-type dataset --revision "${REVISION}" --local-dir "${DEST_DIR}" --max-workers 64

echo "[$(date '+%F %T')] Download finished"
rm -rf "${HF_HUB_CACHE}"
du -sh "${DEST_DIR}"
