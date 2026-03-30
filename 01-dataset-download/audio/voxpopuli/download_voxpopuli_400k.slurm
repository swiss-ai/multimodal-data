#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-voxpopuli
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/voxpopuli-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/voxpopuli-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools
bash /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/voxpopuli/download_voxpopuli_400k.sh
