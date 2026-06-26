#!/bin/bash
# Normalize PangeaIns_train.jsonl image refs to tar-member paths.
# Uses the self-contained normalize_jsonl_tar.py co-located in this directory.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=normalize-pangea
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pangea-instruct-normalize-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pangea-instruct-normalize-%A.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

cd /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/pangea_instruct

OUT_DIR=/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/pangea_instruct

python normalize_jsonl_tar.py \
    --jsonl-in       "$OUT_DIR/PangeaIns_train.jsonl" \
    --tar-root       /capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/pangea_instruct/tars \
    --image-field    image \
    --jsonl-out      "$OUT_DIR/PangeaIns_train_normalized.jsonl" \
    --strip-map-out  "$OUT_DIR/subset_strip_map.json" \
    --drop-log       "$OUT_DIR/dropped.jsonl" \
    --tar-index-workers 32
