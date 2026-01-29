#!/bin/bash
# Download People's Speech dataset (unsupervised) for audio tokenization
#SBATCH --account=infra01
#SBATCH --environment=nemo
#SBATCH --job-name=dwnldHFdset
#SBATCH --output=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/hf-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/hf-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1

export HF_TOKEN="$(cat $HOME/.hf-token)"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/audio-datasets/hf_hub_cache"
export CACHE_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/unsupervised_peoples_speech_cache"
export NUM_PROC=64
export MAX_RETRIES=15
export BACKOFF_FACTOR=1.0

SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/studys/multimodal-data/01-dataset-download"

python "$SCRIPT_DIR/download_hf_dataset.py" \
    --dataset-name "MLCommons/unsupervised_peoples_speech" \
    --cache-dir "$CACHE_DIR" \
    --num-proc "$NUM_PROC" \
    --max-retries "$MAX_RETRIES" \
    --backoff-factor "$BACKOFF_FACTOR"
