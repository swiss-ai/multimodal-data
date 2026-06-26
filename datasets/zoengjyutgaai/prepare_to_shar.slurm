#!/bin/bash
# Cantonese zoengjyutgaai → Lhotse Shar (188h, 4 classic Chinese novels, CC0)
#SBATCH --account=infra01
#SBATCH --job-name=cantonese-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/cantonese-shar-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/cantonese-shar-%x-%A.err
#SBATCH --time=06:00:00
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

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___CanCLID___zoengjyutgaai/data
SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/zoengjyutgaai_train
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json
NUM_WORKERS=128

echo "=========================================="
echo "Cantonese zoengjyutgaai → Shar"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Input: ${RAW_DIR}/*.parquet"
echo "Output: ${SHAR_DIR}"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_DIR}" \
    --parquet-glob '*.parquet' \
    --shar-dir "${SHAR_DIR}" \
    --audio-column audio \
    --text-column transcription \
    --id-column id \
    --duration-column audio_duration \
    --custom-columns episode_id \
    --target-sr 24000 \
    --text-tokenizer "${TOKENIZER}" \
    --shard-size 5000 \
    --num-workers ${NUM_WORKERS}

echo ""
echo "=========================================="
echo "Cantonese zoengjyutgaai done!"
echo "=========================================="
