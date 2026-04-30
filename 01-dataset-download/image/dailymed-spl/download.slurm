#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=dm_spl_dl
#SBATCH --output=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/dm_spl-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/dm_spl-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-10

################################################################################
# DailyMed SPL (Structured Product Labels) — multi-node bulk downloader
################################################################################
# Submits an 11-way SLURM array job. Each array task downloads ONE partition
# ZIP (~5-15 GB each), so 11 nodes download the ~85 GB full release in parallel.
#
# USAGE:
#   sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/dailymed-spl/download.slurm
#
# To download only a subset of partitions, change the #SBATCH --array range
# and/or edit the PARTITIONS array below.
#
# OUTPUT:
#   /capstor/store/cscs/swissai/infra01/medical-datasets/raw/dailymed_spl/
#     └── raw_zips/<partition>.zip      (one per partition)
################################################################################

set -euo pipefail

# Load the `tools` conda env for aria2c (nemo container doesn't ship it).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools

mkdir -p /iopsstor/scratch/cscs/${USER}/multimodal-data/01-dataset-download/logs

PARTITIONS=(
  "dm_spl_release_human_rx_part1"
  "dm_spl_release_human_rx_part2"
  "dm_spl_release_human_rx_part3"
  "dm_spl_release_human_rx_part4"
  "dm_spl_release_human_rx_part5"
  "dm_spl_release_human_otc_part1"
  "dm_spl_release_human_otc_part2"
  "dm_spl_release_human_otc_part3"
  "dm_spl_release_homeopathic"
  "dm_spl_release_animal"
  "dm_spl_release_cellular_therapy"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
if [ "$IDX" -ge "${#PARTITIONS[@]}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$IDX out of range (${#PARTITIONS[@]} partitions)"
  exit 1
fi

PARTITION="${PARTITIONS[$IDX]}"
SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/dailymed-spl"

echo "=========================================="
echo "DailyMed SPL download"
echo "  array task:   ${IDX}"
echo "  partition:    ${PARTITION}"
echo "  job:          ${SLURM_JOB_ID}"
echo "  node:         $(hostname -s)"
echo "  destination:  /capstor/store/cscs/swissai/infra01/medical-datasets/raw/dailymed_spl"
echo "=========================================="

PARTITION="$PARTITION" bash "${SCRIPT_DIR}/download.sh"

echo "=========================================="
echo "Done at $(date)"
echo "=========================================="
