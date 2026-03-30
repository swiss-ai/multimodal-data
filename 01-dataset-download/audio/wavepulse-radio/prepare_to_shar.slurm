#!/bin/bash
# WavePulse Radio → segment MP3s → parquet → Lhotse Shar (~101h US radio speech)
#SBATCH --account=infra01
#SBATCH --job-name=wavepulse-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/wavepulse-shar-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/wavepulse-shar-%x-%A.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

ulimit -n 65536

uv pip install --python /opt/venv/bin/python polars

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}
export FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
export PATH=${FFMPEG_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___nyu-dice-lab___wavepulse-radio-raw-transcripts
PARQUET_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___nyu-dice-lab___wavepulse-radio-raw-transcripts/segmented_parquets
SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/wavepulse_radio_train
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json
SCRIPT_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/wavepulse-radio

echo "=========================================="
echo "Step 1: Segment MP3s → Parquet"
echo "=========================================="

python "${SCRIPT_DIR}/segment_and_convert.py" \
    --recordings-dir "${RAW_DIR}/recordings" \
    --transcripts-dir "${RAW_DIR}/data" \
    --output-dir "${PARQUET_DIR}" \
    --num-workers 32

echo "=========================================="
echo "Step 2: Parquet → Shar"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${PARQUET_DIR}" \
    --parquet-glob 'train-*.parquet' \
    --shar-dir "${SHAR_DIR}" \
    --audio-column audio \
    --text-column text \
    --duration-column duration \
    --custom-columns speaker station state \
    --target-sr 24000 \
    --text-tokenizer "${TOKENIZER}" \
    --shard-size 5000 \
    --num-workers 128

echo ""
echo "=========================================="
echo "WavePulse Radio done!"
echo "=========================================="
