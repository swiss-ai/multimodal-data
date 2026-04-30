#!/bin/bash
# Rsync Voxpopuli SHAR nodes 00-03 from iopsstor to capstor
#SBATCH --account=infra01
#SBATCH --job-name=rsync-voxpopuli
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-3
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/rsync-voxpopuli_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/download/rsync-voxpopuli_%A_%a.err

set -euo pipefail

source /users/xyixuan/miniconda3/bin/activate tools

SRC=/iopsstor/scratch/cscs/xyixuan/audio-datasets/audio_only_shar/voxpopuli_shar
DST=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/voxpopuli_shar

NODE_TAG=$(printf '%02d' ${SLURM_ARRAY_TASK_ID})

echo "===== rsync node_${NODE_TAG} ====="
echo "Start: $(date)"

mkdir -p "${DST}/node_${NODE_TAG}"

rsync -a --whole-file "${SRC}/node_${NODE_TAG}/" "${DST}/node_${NODE_TAG}/"

# Copy shar_index.json on first task
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    cp "${SRC}/shar_index.json" "${DST}/" 2>/dev/null || true
fi

echo "Done: $(date)"
