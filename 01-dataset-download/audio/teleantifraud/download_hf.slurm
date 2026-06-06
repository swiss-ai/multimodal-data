#!/bin/bash
# Download the cleaned Hugging Face release of JimmyMa99/TeleAntiFraud.
#
# This release keeps the payload in a small number of archives:
#   audio.zip, sft.zip, binary_classification.zip, viewer/*.parquet, preview/*.mp3
# Do not unzip here. The processed SFT packer should consume the archives or
# unpack into a temporary staging area and emit self-contained Parquet.
#
# Usage:
#   sbatch download_hf.slurm
#
# HF release inspected on 2026-05-10:
#   https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud
#
# The pinned commit packages the official SFT split:
#   sft/train.jsonl: 27146 records
#   sft/test.jsonl:   6807 records

#SBATCH --account=infra01
#SBATCH --job-name=dwnld-teleantifraud-hf
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/teleantifraud-hf-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/teleantifraud-hf-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --reservation=SD-69241-apertus-1-5

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/git-lfs:${PATH}"

REPO_URL="https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud"
REVISION="0872e54b584b28d34e0911dffcf696f0b2e5e49a"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/sft/hf___JimmyMa99___TeleAntiFraud"
SCRIPT_PATH="01-dataset-download/audio/teleantifraud/download_hf.slurm"

echo "========================================"
echo "TeleAntiFraud HF download"
echo "Repo:     ${REPO_URL}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "========================================"

git lfs install

if [ -d "${DEST_DIR}/.git" ]; then
    echo "Repo already cloned, checking out pinned revision and pulling LFS archives..."
    cd "${DEST_DIR}"
    git fetch origin
    git checkout "${REVISION}"
    git lfs pull
elif [ -d "${DEST_DIR}" ]; then
    echo "Destination exists without .git:"
    echo "  ${DEST_DIR}"
    echo "Assuming this is an already materialized checkout. Remove it to redownload from scratch."
else
    GIT_LFS_SKIP_SMUDGE=1 git clone "${REPO_URL}" "${DEST_DIR}"
    cd "${DEST_DIR}"
    git checkout "${REVISION}"
    git lfs pull
fi

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___JimmyMa99___TeleAntiFraud",
  "repo_url": "${REPO_URL}",
  "revision": "${REVISION}",
  "download_method": "git clone with GIT_LFS_SKIP_SMUDGE=1, git checkout REVISION, git lfs pull",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "git_metadata_removed": true,
  "expected_lfs_objects": {
    "audio.zip": {
      "sha256": "fcc0eef84384e19bda931f23c81b911467aad4ae39af6a69a72788a805953284",
      "size_bytes": 12735906663
    },
    "sft.zip": {
      "sha256": "dfa16c5b4be5b6da44826b691f2e8853578ae1bd5a3e7e55aba8c9167c9032d5",
      "size_bytes": 4390686
    },
    "binary_classification.zip": {
      "sha256": "e54e144fbe4fef41e3f4b8f6ad693a00c7aa3146b751c136a3d7403f0b969585",
      "size_bytes": 60284
    }
  }
}
EOF

rm -rf "${DEST_DIR}/.git"

echo "[$(date '+%F %T')] Done"
echo "Final size:"
du -sh "${DEST_DIR}"
