#!/bin/bash
# Infinity-Doc2-5M prep: split 10 images_labels.tar.gz archives into
# per-subtask (labels.jsonl, images.tar) pairs under processed/.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=prep-infinity-doc
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/infinity-doc2-5m-prep-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/infinity-doc2-5m-prep-%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

cd /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/infinity_doc2_5m
python prep.py --workers 10
