#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=dwnldLN
#SBATCH --output=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/ln-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/ln-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32

################################################################################
# Localized Narratives — audio + text-transcription downloader
################################################################################
# Downloads Google's CC BY 4.0 voice recordings (.ogg) and full annotation
# JSONLs (caption + mouse trace + audio offsets) for all splits.
#
# USAGE:
#   sbatch download.slurm                       # all splits (~200-350 GB)
#   SPLIT_REGEX=open_images sbatch download.slurm   # OpenImages only
#   SPLIT_REGEX=coco_val   sbatch download.slurm    # smoke test (~3 GB)
#
# OUTPUT:
#   /capstor/store/cscs/swissai/infra01/audio-datasets/raw/localized_narratives/
#     ├── annotations/<split>_*.jsonl
#     └── voice-recordings/<split>/*.ogg
################################################################################

set -euo pipefail

# Load the `tools` conda env for aria2c (nemo container doesn't ship it).
# Matches convention used by audiocite / mediaspeech / hui-audio-corpus-german.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools

mkdir -p /iopsstor/scratch/cscs/${USER}/multimodal-data/01-dataset-download/logs

SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/localized_narratives"
SPLIT_REGEX="${SPLIT_REGEX:-.*}"

echo "=========================================="
echo "Localized Narratives audio + text download"
echo "  job:          ${SLURM_JOB_ID}"
echo "  node:         $(hostname -s)"
echo "  split regex:  ${SPLIT_REGEX}"
echo "  destination:  /capstor/store/cscs/swissai/infra01/audio-datasets/raw/localized_narratives"
echo "=========================================="

bash "${SCRIPT_DIR}/download.sh" "${SPLIT_REGEX}"

echo "=========================================="
echo "Done at $(date)"
echo "=========================================="
