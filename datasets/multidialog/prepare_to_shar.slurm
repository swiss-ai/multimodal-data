#!/bin/bash
# MultiDialog → Lhotse Shar (spoken dialogue, ~253h, 7K conversations)
#SBATCH --account=infra01
#SBATCH --job-name=multidialog-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/multidialog-shar-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/multidialog-shar-%x-%A.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

ulimit -n 65536

uv pip install --python /opt/venv/bin/python orjson

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___IVLLab___MultiDialog
SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/multidialog_train
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

echo "=========================================="
echo "MultiDialog → Shar"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Input: ${RAW_DIR}/data/train/*.tar.gz"
echo "Metadata: ${RAW_DIR}/metadata.jsonl"
echo "Output: ${SHAR_DIR}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards "${RAW_DIR}/data/train/*.tar.gz" \
    --external-metadata "${RAW_DIR}/metadata.jsonl" \
    --id-field file_name \
    --text-field value \
    --custom-fields conv_id utterance_id from emotion \
    --shar-dir "${SHAR_DIR}" \
    --target-sr 24000 \
    --shard-size 5000 \
    --shar-format flac \
    --text-tokenizer "${TOKENIZER}" \
    --num-workers 32

echo ""
echo "=========================================="
echo "MultiDialog done!"
echo "=========================================="
