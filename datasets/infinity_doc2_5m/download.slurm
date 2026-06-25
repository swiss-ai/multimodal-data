#!/bin/bash
# Download infly/Infinity-Doc2-5M (~125 GiB, MIT, en+zh, ~5M document-parsing samples).
# 10 tar.gz shards across 6 task categories:
#   Blank_Page_Parsing / Chart_Parsing (chart2code, chart2json, chart2table)
#   Chemical_Formula_Parsing (chem2smiles) / Document_VQA (docvqa)
#   Element_Parsing (formula2latex, table2html, table2md) / Layout_Analysis
# Directly fills the table-image SFT gap (table2html, table2md, chart2table).
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/infinity_doc2_5m/download.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-infinity-doc2-5m
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/infinity-doc2-5m-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/infinity-doc2-5m-%x-%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

DATASET_REPO="infly/Infinity-Doc2-5M"
REVISION="993ca58e4e935d767b4f263aff68f9d0eafbe82a"
DEST_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___infly___Infinity-Doc2-5M"
SCRIPT_PATH="01-dataset-download/image/infinity_doc2_5m/download.slurm"

export HF_HUB_CACHE="${DEST_DIR}/.hf_cache"

mkdir -p "${DEST_DIR}"

echo "========================================"
echo "Infinity-Doc2-5M Download"
echo "Repo:     ${DATASET_REPO}"
echo "Revision: ${REVISION}"
echo "Dest:     ${DEST_DIR}"
echo "Expected: ~125 GiB, MIT, 10 tar.gz shards + parquet preview"
echo "========================================"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers 64

echo "[$(date '+%F %T')] Download finished"

cat > "${DEST_DIR}/_SOURCE.json" <<EOF
{
  "dataset_id": "hf___infly___Infinity-Doc2-5M",
  "repo_url": "https://huggingface.co/datasets/${DATASET_REPO}",
  "revision": "${REVISION}",
  "license": "mit",
  "languages": ["en", "zh"],
  "download_method": "huggingface-cli download with HF_HUB_ENABLE_HF_TRANSFER=1",
  "downloaded_at_utc": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "script": "${SCRIPT_PATH}",
  "size_gib": 125,
  "notes": "Document-parsing SFT pack, ~5M samples across 6 task categories. Each subset ships images_labels.tar.gz that needs extraction before training. Strong table/chart/formula coverage — fills the table-image SFT gap. demo_data/ has a small parquet preview."
}
EOF

rm -rf "${HF_HUB_CACHE}"

echo "Final size:"
du -sh "${DEST_DIR}"
echo "[$(date '+%F %T')] Done"
