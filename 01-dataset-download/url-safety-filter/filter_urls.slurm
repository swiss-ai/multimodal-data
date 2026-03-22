#!/bin/bash
# Pre-filter URLs in parquet datasets against cybersecurity blocklists
#SBATCH --account=infra01
#SBATCH --job-name=url-safety-filter
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/url-safety-filter-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/url-safety-filter-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Configure these per dataset ----
INPUT_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/OpenFace-CQUPT___FaceCaption-15M"
OUTPUT_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/OpenFace-CQUPT___FaceCaption-15M_safe"
URL_COLUMN="url"
# --------------------------------------

echo "========================================"
echo "URL Safety Filter"
echo "Input:      ${INPUT_DIR}"
echo "Output:     ${OUTPUT_DIR}"
echo "URL column: ${URL_COLUMN}"
echo "========================================"

python "${SCRIPT_DIR}/filter_unsafe_urls.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --url-column "${URL_COLUMN}" \
    --workers 64 \
    --save-blocklist "${OUTPUT_DIR}/blocked_domains.txt"

echo "[$(date '+%F %T')] Done."
