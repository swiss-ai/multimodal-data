#!/bin/bash
# Download Harland/AudioMCQ-StrongAC-GeminiCoT (curated AudioMCQ subset with
# Gemini 3.1 Pro CoT reasoning; 7 source families: AudioCaps, Clotho, CompA-R,
# LP-MusicCaps-MTT, MusicCaps, SpeechCraft, Tacos; ~tens of thousands of WAVs +
# data.jsonl, Apache-2.0).
# Uses git clone + git lfs pull because this dataset has many individual audio
# files and the Hugging Face CLI path is much slower for that access pattern.
#
# Usage:
#   sbatch download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-audiomcq-cot
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/audiomcq-cot-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/audiomcq-cot-%x-%A.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PATH=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/git-lfs:${PATH}
git lfs install

REPO_URL="https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT"
REVISION="cb33687ce4dc3dace4e203a2dd584fa46ae312da"
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/sft/hf___Harland___AudioMCQ-StrongAC-GeminiCoT}
DROP_GIT_METADATA=${DROP_GIT_METADATA:-1}

echo "========================================"
echo "AudioMCQ-StrongAC-GeminiCoT Download (git clone + LFS)"
echo "Repo:     ${REPO_URL}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

if [ ! -d "${DEST_DIR}/.git" ] && [ -f "${DEST_DIR}/_SOURCE.json" ] && [ -f "${DEST_DIR}/data.jsonl" ]; then
    echo "Materialized snapshot already exists without Git metadata."
    echo "Set DROP_GIT_METADATA=0 and reclone to refresh/resume via Git LFS."
elif [ -d "${DEST_DIR}/.git" ]; then
    echo "Repo already cloned, pulling remaining LFS files..."
    cd "${DEST_DIR}"
    git checkout "${REVISION}"
    git lfs pull
else
    if [ -d "${DEST_DIR}" ] && [ -n "$(find "${DEST_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
        echo "Destination exists but is not a Git checkout or completed snapshot: ${DEST_DIR}" >&2
        exit 1
    fi
    GIT_LFS_SKIP_SMUDGE=1 git clone \
        "${REPO_URL}" \
        "${DEST_DIR}"
    cd "${DEST_DIR}"
    git checkout "${REVISION}"
    git lfs pull
fi

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "repo_id": "Harland/AudioMCQ-StrongAC-GeminiCoT",
  "repo_type": "dataset",
  "revision": "${REVISION}",
  "source_url": "${REPO_URL}",
  "downloaded_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "download_script": "$0",
  "license_note": "Dataset card is Apache-2.0; embedded audio has mixed upstream provenance."
}
EOF

if [ "${DROP_GIT_METADATA}" = "1" ] && [ -d "${DEST_DIR}/.git" ]; then
    echo "Dropping Git metadata after successful LFS materialization..."
    rm -rf "${DEST_DIR}/.git"
fi

echo "[$(date '+%F %T')] Done"
echo "Pinned to commit: ${REVISION}"
echo "Final size:"
du -sh "${DEST_DIR}"
