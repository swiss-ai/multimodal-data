#!/bin/bash
# Download Multilingual LibriSpeech dataset for audio tokenization
#SBATCH --account=infra01
#SBATCH --environment=nemo
#SBATCH --job-name=dwnld-mls
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/hf-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/hf-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --reservation=PA-2338-RL
#SBATCH --nodes=1

export HF_TOKEN="$(cat $HOME/.hf-token)"

# Install hf_transfer for faster downloads
pip install -q hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/audio-datasets/hf_hub_cache"
export CACHE_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/mls_cache"
export NUM_PROC=64
export MAX_RETRIES=15
export BACKOFF_FACTOR=1.0

SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download"

# Download all 7 non-English subsets of Multilingual LibriSpeech
# Subsets: german, dutch, french, spanish, italian, portuguese, polish
# Each subset has splits: train, dev, test, 9_hours, 1_hours

python "$SCRIPT_DIR/download_hf_dataset.py" \
    --dataset-name "facebook/multilingual_librispeech" \
    --subset-name "german,dutch,french,spanish,italian,portuguese,polish" \
    --cache-dir "$CACHE_DIR" \
    --num-proc "$NUM_PROC" \
    --max-retries "$MAX_RETRIES" \
    --backoff-factor "$BACKOFF_FACTOR"
