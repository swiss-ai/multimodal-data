#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=suno-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/suno/logs/suno_shar_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/suno/logs/suno_shar_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/suno/logs

echo "Suno shar conversion | Job: ${SLURM_JOB_ID} | Node: $(hostname)"

python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards '/capstor/store/cscs/swissai/infra01/audio-datasets/raw/suno/suno_S1/shards/*.tar' \
    --shar-dir /iopsstor/scratch/cscs/xyixuan/audio-datasets/audio_only_shar/suno_train_shar \
    --target-sr 24000 \
    --min-sr 16000 \
    --shard-size 5000 \
    --shar-format flac \
    --text-ext '' \
    --num-workers 64

echo "Suno shar conversion finished"
