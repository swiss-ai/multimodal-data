#!/bin/bash
# Rsync UPS SHAR nodes 05-26 from iopsstor to capstor
# No --environment: bare metal has rsync
#SBATCH --account=infra01
#SBATCH --job-name=rsync-ups
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=5-26
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/rsync-ups_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/rsync-ups_%A_%a.err

set -euo pipefail

source /users/xyixuan/miniconda3/bin/activate tools

SRC=/iopsstor/scratch/cscs/xyixuan/audio-datasets/audio_only_shar/ups_all_lang_shar_v2
DST=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/ups_all_lang_shar_v2

NODE_TAG=$(printf '%02d' ${SLURM_ARRAY_TASK_ID})

echo "===== rsync node_${NODE_TAG} ====="
echo "Start: $(date)"
echo "From: ${SRC}/node_${NODE_TAG}/"
echo "To:   ${DST}/node_${NODE_TAG}/"

mkdir -p "${DST}/node_${NODE_TAG}"

rsync -a --whole-file "${SRC}/node_${NODE_TAG}/" "${DST}/node_${NODE_TAG}/"

echo "Done: $(date)"
