#!/bin/bash
# Download RUC-NLPIR/Omnimodal-Agent-SFT-2K (~27GB, 6028 files)
#SBATCH --account=infra01
#SBATCH --job-name=dl-omnimodal_ag
#SBATCH --reservation=SD-69241-apertus-1-5-3
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/omnimodal_agent_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/omnimodal_agent_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

export PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/git-lfs:${PATH}
git lfs install

REPO_URL="https://huggingface.co/datasets/RUC-NLPIR/Omnimodal-Agent-SFT-2K"
REVISION="067e71b09bec"
DEST="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___RUC-NLPIR___Omnimodal-Agent-SFT-2K"

echo "=========================================="
echo "Downloading: ${REPO_URL}"
echo "Revision: ${REVISION}"
echo "Dest: ${DEST}"
echo "=========================================="

GIT_LFS_SKIP_SMUDGE=1 git clone "${REPO_URL}" "${DEST}"
cd "${DEST}"
git checkout "${REVISION}"
git lfs pull

echo "Done: omnimodal_agent"
