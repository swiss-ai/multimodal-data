#!/bin/bash
# Download MRSAudio/MRSAudio from HuggingFace via git clone + LFS
# 500h multimodal spatial audio, 246K segments, CC-BY-4.0
# git clone batches LFS downloads — much faster for many small files
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-mrsaudio
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mrsaudio-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mrsaudio-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

export PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/git-lfs:${PATH}
git lfs install

DATASET_REPO="MRSAudio/MRSAudio"
REVISION="b1349c4fc03c11b958c2d0e5deb62ae208c1a43c"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___MRSAudio___MRSAudio"
CLONE_DIR="${DEST_DIR}_git_clone"

echo "========================================"
echo "MRSAudio Download (git clone + LFS)"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Clone to: ${CLONE_DIR}"
echo "========================================"

if [ -d "${CLONE_DIR}/.git" ]; then
    echo "Repo already cloned, pulling remaining LFS files..."
    cd "${CLONE_DIR}"
    git lfs pull
else
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        "https://huggingface.co/datasets/${DATASET_REPO}" \
        "${CLONE_DIR}"
    cd "${CLONE_DIR}"
    git checkout "${REVISION}"
    git lfs pull
fi

echo "[$(date '+%F %T')] LFS pull finished"

# Move data to final destination (merge with existing MRSLife)
for subset in MRSLife MRSMusic MRSSing MRSSpeech; do
    if [ -d "${CLONE_DIR}/${subset}" ] && [ ! -d "${DEST_DIR}/${subset}" ]; then
        echo "Moving ${subset} to ${DEST_DIR}/"
        mv "${CLONE_DIR}/${subset}" "${DEST_DIR}/"
    elif [ -d "${CLONE_DIR}/${subset}" ]; then
        echo "${subset} already exists in dest, skipping"
    fi
done

# Copy metadata files
cp -n "${CLONE_DIR}/train.csv" "${DEST_DIR}/" 2>/dev/null || true
cp -n "${CLONE_DIR}/README.md" "${DEST_DIR}/" 2>/dev/null || true

echo "[$(date '+%F %T')] Done"
echo "Final size:"
du -sh "${DEST_DIR}"
