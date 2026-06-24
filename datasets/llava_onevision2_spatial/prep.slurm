#!/bin/bash
# Reshard LOV2 spatial osd + roborefer_sim into processed/sft jsonl_tar format.
#
# Usage:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/llava_onevision2_spatial/prep.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=reshard-lov2-spatial
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/reshard-lov2-spatial-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/reshard-lov2-spatial-%x-%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"

SCRIPT=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/llava_onevision2_spatial/prep.py

echo "[$(date '+%F %T')] reshard start"
python3 "$SCRIPT" --workers 19
echo "[$(date '+%F %T')] reshard done"

echo "--- output ---"
du -sh /capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/hf___mvp-lab___LLaVA-OneVision-2-Data___osd
du -sh /capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/hf___mvp-lab___LLaVA-OneVision-2-Data___roborefer_sim
