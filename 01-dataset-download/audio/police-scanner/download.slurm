#!/bin/bash
# Download trentmkelly/police-scanner-audio (~357GB, 50k hours, 6 US cities, CC-BY-3.0)
# Uses git clone + git lfs pull (much faster for 103k small MP3 files).
#
# Usage:
#   sbatch download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-police-scanner
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/police-scanner-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/police-scanner-%x-%A.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/git-lfs:${PATH}
git lfs install

REPO_URL="https://huggingface.co/datasets/trentmkelly/police-scanner-audio"
REVISION="302d8babf66efc37320407beed267ee7672b3d51"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___trentmkelly___police-scanner-audio"

echo "========================================"
echo "Police Scanner Download (git clone + LFS)"
echo "Repo:     ${REPO_URL}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

if [ -d "${DEST_DIR}/.git" ]; then
    echo "Repo already cloned, pulling remaining LFS files..."
    cd "${DEST_DIR}"
    git lfs pull
else
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        "${REPO_URL}" \
        "${DEST_DIR}"
    cd "${DEST_DIR}"
    git checkout "${REVISION}"
    git lfs pull
fi

echo "[$(date '+%F %T')] Done"
echo "Final size:"
du -sh "${DEST_DIR}"
