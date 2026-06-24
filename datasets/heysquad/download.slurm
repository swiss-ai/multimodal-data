#!/bin/bash
# Download HeySQuAD human + machine (spoken QA, CC-BY-4.0)
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-heysquad
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/heysquad-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/heysquad-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "HeySQuAD Download (human + machine)"
echo "========================================"

# Human
DEST_H="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___yijingwu___HeySQuAD_human"
export HF_HUB_CACHE="${DEST_H}/.hf_cache"
mkdir -p "${DEST_H}"
huggingface-cli download "yijingwu/HeySQuAD_human" --repo-type dataset --revision "2a9aff6032a76185b98eff703144a2a896313e49" --local-dir "${DEST_H}" --max-workers 64
rm -rf "${HF_HUB_CACHE}"
echo "[$(date '+%F %T')] Human done"

# Machine
DEST_M="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___yijingwu___HeySQuAD_machine"
export HF_HUB_CACHE="${DEST_M}/.hf_cache"
mkdir -p "${DEST_M}"
huggingface-cli download "yijingwu/HeySQuAD_machine" --repo-type dataset --revision "129220a5a6d27eb2c70be4ecc2e4bbcad4cca436" --local-dir "${DEST_M}" --max-workers 64
rm -rf "${HF_HUB_CACHE}"
echo "[$(date '+%F %T')] Machine done"

echo "Done."
