#!/bin/bash
# Download AISHELL-1, AISHELL-3, AISHELL-4 from OpenSLR via Lhotse
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-aishell
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/aishell-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/aishell-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

BASE_DIR=${BASE_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw}

echo "========================================"
echo "AISHELL-1/3/4 Download via Lhotse"
echo "Base: ${BASE_DIR}"
echo "========================================"

# AISHELL-1: lhotse extracts into target_dir/aishell/ automatically
echo "[$(date '+%F %T')] Downloading AISHELL-1 (~178h, OpenSLR 33)..."
mkdir -p "${BASE_DIR}/aishell1"
lhotse download aishell "${BASE_DIR}/aishell1"

# AISHELL-3: extracts into target_dir/ directly
echo "[$(date '+%F %T')] Downloading AISHELL-3 (~85h, OpenSLR 93)..."
mkdir -p "${BASE_DIR}/aishell3"
lhotse download aishell3 "${BASE_DIR}/aishell3"

# AISHELL-4: extracts train_L/ train_M/ train_S/ test/ into target_dir/
echo "[$(date '+%F %T')] Downloading AISHELL-4 (~120h, OpenSLR 111)..."
mkdir -p "${BASE_DIR}/aishell4"
lhotse download aishell4 "${BASE_DIR}/aishell4"

echo "[$(date '+%F %T')] All downloads finished"
echo "Done."
