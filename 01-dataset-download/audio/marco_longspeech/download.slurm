#!/bin/bash
# Download AIDC-AI/Marco_Longspeech (~2076 GB, 103876 wav files, Apache 2.0)
# EN/ZH long-form speech understanding, 8 tasks.
# Uses git clone + git lfs pull (much faster for 100k+ small files).
#
# Usage:
#   sbatch download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-marco-longspeech
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/marco-longspeech-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/marco-longspeech-%x-%A.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/git-lfs:${PATH}
git lfs install

REPO_URL="https://huggingface.co/datasets/AIDC-AI/Marco_Longspeech"
REVISION="e72c4a1fb453225ffdd97c5d2c0c832a367cf6d8"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/sft/hf___AIDC-AI___Marco_Longspeech"

echo "========================================"
echo "Marco_Longspeech Download (git clone + LFS)"
echo "Repo:     ${REPO_URL}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

if [ -d "${DEST_DIR}/.git" ]; then
    echo "Repo already cloned, pulling remaining LFS files..."
    cd "${DEST_DIR}"
    git checkout "${REVISION}"
    git lfs pull
elif [ -d "${DEST_DIR}" ]; then
    echo "Destination exists without .git:"
    echo "  ${DEST_DIR}"
    echo "Assuming this is an already materialized checkout. Remove the directory to redownload from scratch."
else
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        "${REPO_URL}" \
        "${DEST_DIR}"
    cd "${DEST_DIR}"
    git checkout "${REVISION}"
    git lfs pull
fi

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "repo_url": "${REPO_URL}",
  "revision": "${REVISION}",
  "download_method": "git clone with GIT_LFS_SKIP_SMUDGE=1, git checkout REVISION, git lfs pull",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "git_metadata_removed": true,
  "script": "01-dataset-download/audio/marco_longspeech/download.slurm"
}
EOF

rm -rf "${DEST_DIR}/.git"

echo "[$(date '+%F %T')] Done"
echo "Final size:"
du -sh "${DEST_DIR}"
