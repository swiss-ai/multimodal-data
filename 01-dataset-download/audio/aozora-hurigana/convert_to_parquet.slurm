#!/bin/bash
# Convert Aozora Hurigana Speech v2 nested zips → parquet (in-memory, no extraction)
# 288 CPUs parallel processing of author zips
#SBATCH --account=infra01
#SBATCH --job-name=aozora-to-parquet
#SBATCH --reservation=SD-69241-apertus-1-5-6
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/aozora-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/aozora-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

uv pip install --python /opt/venv/bin/python polars

INPUT_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/ndl___aozora_hurigana_speech_v2"
OUTPUT_DIR="${INPUT_DIR}/parquet"

echo "========================================"
echo "Aozora Hurigana → Parquet"
echo "Input:   ${INPUT_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "Workers: 256"
echo "========================================"

/opt/venv/bin/python /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/aozora-hurigana/convert_to_parquet.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers 256 \
    --samples-per-parquet 50000

echo "[$(date '+%F %T')] Done"
