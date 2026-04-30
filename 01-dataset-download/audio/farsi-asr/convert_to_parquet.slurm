#!/bin/bash
# Convert farsi-asr tar.gz archives → parquet (in-memory, no extraction)
#SBATCH --account=infra01
#SBATCH --job-name=farsi-asr-to-parquet
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/farsi-asr-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/farsi-asr-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

INPUT_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___farsi-asr___farsi-asr-dataset"
OUTPUT_DIR="${INPUT_DIR}/parquet"

echo "========================================"
echo "Farsi ASR → Parquet"
echo "Input:   ${INPUT_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "Workers: 288"
echo "========================================"

/opt/venv/bin/python /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/farsi-asr/convert_to_parquet.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers 288

echo "[$(date '+%F %T')] Done"
