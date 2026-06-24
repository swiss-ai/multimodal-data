#!/bin/bash
# Download KurtDu/EchoX-Dialogues-Plus, S2T-QA split only (~308 GB, 66 tar.gz, Apache 2.0).
# S2T-QA = Speech-to-Text question answering dialogues (Magpie-Pro-Speech-ALL+).
# Texts are public, synthesized output speech. Uses huggingface-cli (few large files).
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/echox_dialogues_plus/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-echox-s2t
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/echox-dialogues-plus-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/echox-dialogues-plus-%x-%A.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="KurtDu/EchoX-Dialogues-Plus"
REVISION="d4394d50ba54d8c42d1f5ef9f4cef2d12719dabf"
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/sft/hf___KurtDu___EchoX-Dialogues-Plus"
SCRIPT_PATH="01-dataset-download/audio/echox_dialogues_plus/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "EchoX-Dialogues-Plus S2T-QA Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Filter:   S2T-QA/* + README.md"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --include "S2T-QA/*" "README.md" \
    --max-workers 64

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___KurtDu___EchoX-Dialogues-Plus",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "subset": "S2T-QA/Magpie-Pro-Speech-ALL+",
  "download_method": "huggingface-cli download with --include S2T-QA/*, HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "notes": "S2T-QA split only (Speech-to-Text QA, Magpie-Pro-Speech-ALL+, 66 tar.gz shards). S2S-QA/AudioQA not downloaded."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
