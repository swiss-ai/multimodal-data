#!/bin/bash
# Reassemble Pangea split-part tars per upstream HF README convention:
#   cat part_* > images.tar
# Affects: general/ALLaVA-4V/allava_laion.tar (2 parts) + cultural/laion-multi-1M/images.tar (4 parts).
# Recovers ~2M records that the first normalize pass dropped because these
# archives weren't reassembled.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=cat-pangea-parts
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pangea-instruct-cat-parts-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pangea-instruct-cat-parts-%A.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

cd /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/pangea_instruct
python prep.py --steps cat_parts --cat-workers 2
