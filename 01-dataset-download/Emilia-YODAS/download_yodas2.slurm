#!/bin/bash
# Download YODAS2 dataset from HuggingFace (espnet/yodas2)
#SBATCH --account=infra01
#SBATCH --environment=nemo
#SBATCH --job-name=dwnldYODAS2
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/hf-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/hf-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

DATASET_REPO=${DATASET_REPO:-espnet/yodas2}
# Default: download all languages. Override to filter, e.g. "data/en000/*" or "data/de000/*"
INCLUDE_PATTERN=${INCLUDE_PATTERN:-data/*}
DEST_DIR=${DEST_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/yodas2}
CPUS_AVAILABLE=${SLURM_CPUS_PER_TASK:-$(nproc)}
MAX_WORKERS=${MAX_WORKERS:-$CPUS_AVAILABLE}
LOG_DIR=${LOG_DIR:-/iopsstor/scratch/cscs/${USER}/apertus/multimodal-data/01-dataset-download/logs}

mkdir -p "${LOG_DIR}"
mkdir -p "${DEST_DIR}"

export HF_HUB_ENABLE_HF_TRANSFER=1

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found in PATH"
  exit 1
fi

echo "[$(date '+%F %T')] Download start"
echo "Repo: ${DATASET_REPO}"
echo "Include: ${INCLUDE_PATTERN}"
echo "Destination: ${DEST_DIR}"
echo "Workers: ${MAX_WORKERS}"

huggingface-cli download "${DATASET_REPO}" \
  --repo-type dataset \
  --include "${INCLUDE_PATTERN}" \
  --local-dir "${DEST_DIR}" \
  --max-workers "${MAX_WORKERS}"

echo "[$(date '+%F %T')] Download finished"
