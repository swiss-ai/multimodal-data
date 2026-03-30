#!/bin/bash
# Filter webdataset shards to commercially-licensed samples only
#SBATCH --account=infra01
#SBATCH --environment=nemo
#SBATCH --job-name=filterWDS
#SBATCH --output=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/filter-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/multimodal-data/01-dataset-download/logs/filter-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

SNAPSHOT_DIR="/capstor/store/cscs/swissai/infra01/audio-datasets/hf_hub_cache/datasets--MLCommons--unsupervised_peoples_speech/snapshots/d917e17e86f6abc4fa4d83e958c8f4173f45f0e7"
ALLOWED_IDS="/capstor/store/cscs/swissai/infra01/audio-datasets/peoples_speech_commercial_ids.txt"
OUTPUT_DIR="/iopsstor/scratch/cscs/xyixuan/audio-datasets/unsupervised_peoples_speech_commercial_wds"
NUM_WORKERS=288

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

python "$SCRIPT_DIR/license_filter.py" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    --allowed-ids "$ALLOWED_IDS" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers "$NUM_WORKERS"
