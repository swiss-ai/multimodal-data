#!/bin/bash
#SBATCH --job-name=mathnet
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=288
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mathnet-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mathnet-%j.err

# ShadenA/MathNet
#   https://huggingface.co/datasets/ShadenA/MathNet
#
# Multimodal olympiad math corpus. 27.8K rows, 17 languages, problem + solution
# pairs with embedded figures (LaTeX markdown + images).
#
# License: CC-BY-4.0 — permissive for commercial training with attribution.
#
# Role: SFT material (multi-step reasoning + multimodal math). Can also slot
# 1-3% into stage-2 visual instruction pretraining for early math-diagram
# exposure. Scale is too small (~50M tokens) to meaningfully shift main-stage
# pretraining; concentrate in SFT / stage-2 where the high-quality signal
# isn't diluted.
#
# Strategy: huggingface-cli download with HF_TRANSFER — parquet files with
# embedded image bytes, few-but-large → hf_transfer parallel chunking wins.

set -euo pipefail

DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/mathnet
mkdir -p "$DEST"

# hf_transfer lives in the shared pip-packages dir (matches nemotron-vlm,
# fine-t2i, etc.). Keeps it out of the nemo container.
export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=600

echo "=== downloading ShadenA/MathNet to $DEST ==="
date -u +%Y-%m-%dT%H:%M:%SZ

time huggingface-cli download ShadenA/MathNet \
    --repo-type dataset \
    --local-dir "$DEST"

echo
echo "=== final summary ==="
du -sh "$DEST"
find "$DEST" -maxdepth 2 -name "*.parquet" | sort | head -10
date -u +%Y-%m-%dT%H:%M:%SZ
