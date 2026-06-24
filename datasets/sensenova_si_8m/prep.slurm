#!/bin/bash
# SenseNova-SI-8M prep: re-tar 52 STORE-mode image zips → uncompressed tars.
# Single 1-node CPU job, parallel xargs across zips.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=prep-sensenova
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/sensenova-si-8m-prep-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/sensenova-si-8m-prep-%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

cd /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/sensenova_si_8m

echo "==========================================================="
echo "SenseNova-SI-8M prep — re-tar 52 zips (STORE mode → tar)"
echo "==========================================================="

python prep.py --retar-workers 32

echo
echo "==========================================================="
echo "Done. Artifacts: 52 .tar files alongside images_part_*.zip"
echo "==========================================================="
