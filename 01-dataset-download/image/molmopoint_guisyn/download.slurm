#!/bin/bash
# Download allenai/MolmoPoint-GUISyn (~14 GiB, ODC-BY, en, ~30K GUI screenshots).
# Synthetic GUI grounding / pointing SFT dataset for Molmo / MolmoPoint.
# 3 configs: desktop (18 train shards), web (11), mobile (4); each has 1 validation shard.
# Each example = screenshot + ~54 (intent -> point) annotations with bbox + HTML metadata.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/molmopoint_guisyn/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-molmopoint-guisyn
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/molmopoint-guisyn-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/molmopoint-guisyn-%x-%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="allenai/MolmoPoint-GUISyn"
REVISION="24bb3e990bb1e48796d80ade9dbc858dda695e75"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___allenai___MolmoPoint-GUISyn"
SCRIPT_PATH="01-dataset-download/image/molmopoint_guisyn/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "MolmoPoint-GUISyn Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Expected: ~14 GiB, ODC-BY, 38 parquet shards across desktop/web/mobile + paper"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 64

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___allenai___MolmoPoint-GUISyn",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "odc-by",
  "license_notes": "ODC-BY + Ai2 Responsible Use + Anthropic ToS (synthetic outputs from Claude Sonnet 4.6). Permissive for research; commercial use inherits Anthropic output-use terms.",
  "languages": ["en"],
  "data_type": "sft",
  "task": "gui_grounding_pointing",
  "download_method": "huggingface-cli download with HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "size_gib": 14,
  "notes": "GUI grounding / pointing SFT for Molmo / MolmoPoint (arxiv 2603.28069). Configs: desktop (16232+256), web (9830+256), mobile (subset). ~54 (name + 5 intents + bbox + center) annotations per image; HTML metadata also included. Pointing-head SFT, not general image SFT."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
