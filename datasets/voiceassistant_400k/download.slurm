#!/bin/bash
# Download gpt-omni/VoiceAssistant-400K from Hugging Face
# 470K rows, parquet format, contains question_audio + answer SNAC codes
#SBATCH --account=infra01
#SBATCH --environment=nemo
#SBATCH --job-name=dwnld-va400k
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/va400k-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/va400k-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL


set -euo pipefail

DATASET_REPO="gpt-omni/VoiceAssistant-400K"
REVISION="65dab707fcdd3d43dcfb834398aa9fed4116be3a"  # 2024-09-13, 470K rows
DEST_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/VoiceAssistant-400K"
CPUS_AVAILABLE=${SLURM_CPUS_PER_TASK:-$(nproc)}
MAX_WORKERS=${MAX_WORKERS:-$CPUS_AVAILABLE}

mkdir -p "${DEST_DIR}"

uv pip install --system --break-system-packages hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "ERROR: huggingface-cli not found in PATH"
    exit 1
fi

echo "========================================"
echo "VoiceAssistant-400K Download"
echo "========================================"
echo "[$(date '+%F %T')] Download start"
echo "Repo: ${DATASET_REPO}"
echo "Destination: ${DEST_DIR}"
echo "Workers: ${MAX_WORKERS}"

huggingface-cli download "${DATASET_REPO}" \
    --repo-type dataset \
    --revision "${REVISION}" \
    --local-dir "${DEST_DIR}" \
    --max-workers "${MAX_WORKERS}"

echo "[$(date '+%F %T')] Download finished"
