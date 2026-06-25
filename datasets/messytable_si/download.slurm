#!/bin/bash
# Download sensenova/MessyTable-SI (~22 GiB, Apache-2.0, English spatial QA).
# Multi-view table scene understanding: object identification, horizontal rotation,
# same-scene matching — with chain-of-thought variants (wcot).
# Split zip images (images.zip.001-006) need: cat images.zip.* > images.zip && unzip.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/messytable_si/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-messytable-si
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/messytable-si-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/messytable-si-%x-%A.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="sensenova/MessyTable-SI"
REVISION="69bd2371e418a3fea6d36d8e09f101fbe895b071"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___sensenova___MessyTable-SI"
SCRIPT_PATH="01-dataset-download/image/messytable_si/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "MessyTable-SI Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Expected: ~22 GiB, Apache-2.0, 15 files"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 32

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___sensenova___MessyTable-SI",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "apache-2.0",
  "data_type": "sft",
  "task": "spatial_qa_table_scene",
  "download_method": "huggingface-cli download with HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "size_gib": 22,
  "notes": "Multi-view table scene spatial understanding (SenseNova SI series). Split zip: cat images.zip.001-006 > images.zip && unzip. Labels: labels.zip + 4 JSONL train sets (10k each, with/without CoT variants). Companion to SenseNova-SI-8M."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
