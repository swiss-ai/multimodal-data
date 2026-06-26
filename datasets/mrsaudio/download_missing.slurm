#!/bin/bash
# Re-download missing MRSAudio subsets via git lfs pull
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-mrsaudio
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mrsaudio-missing-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mrsaudio-missing-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

export PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/git-lfs:${PATH}
git lfs install

CLONE_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___MRSAudio___MRSAudio_git_clone
DEST_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___MRSAudio___MRSAudio

echo "========================================"
echo "MRSAudio — pull missing subsets"
echo "========================================"

cd "${CLONE_DIR}"
git lfs pull --include="MRSMusic/**,MRSSing/**,MRSSpeech/**"

echo "[$(date '+%F %T')] LFS pull done"
echo "Counts:"
for s in MRSLife MRSMusic MRSSing MRSSpeech; do
    echo "  ${s}: $(ls ${CLONE_DIR}/${s}/ 2>/dev/null | wc -l) dirs"
done

# Rename git clone to final destination
mv "${CLONE_DIR}" "${DEST_DIR}"
echo "[$(date '+%F %T')] Renamed to ${DEST_DIR}"
