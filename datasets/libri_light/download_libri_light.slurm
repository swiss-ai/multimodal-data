#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=dwnld-librilight
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/librilight-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/librilight-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tools
bash /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/libri-light/download_libri_light.sh all
