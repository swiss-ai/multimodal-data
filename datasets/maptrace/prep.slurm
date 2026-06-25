#!/bin/bash
# Reshape google/MapTrace raw parquet → SFT-shaped parquet under processed/.
# Drops the eval split (maptrace_20k); processes ~5,336 train shards (~216 GB).
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=prep-maptrace
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/maptrace-prep-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/maptrace-prep-%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail
cd /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/maptrace
python prep.py --workers 64
