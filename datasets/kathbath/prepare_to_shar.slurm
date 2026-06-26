#!/bin/bash
# Kathbath → Lhotse Shar (1684h, 12 Indian languages, read speech, FLAC)
#SBATCH --account=infra01
#SBATCH --job-name=kathbath-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/kathbath-shar-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/kathbath-shar-%x-%A.err
#SBATCH --time=04:00:00
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

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___ai4bharat___Kathbath
SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/kathbath_train
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

echo "=========================================="
echo "Kathbath → Shar"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_DIR}" \
    --parquet-glob '**/*.parquet' \
    --shar-dir "${SHAR_DIR}" \
    --audio-column audio_filepath \
    --text-column text \
    --duration-column duration \
    --language-column lang \
    --custom-columns gender speaker_id \
    --target-sr 24000 \
    --text-tokenizer "${TOKENIZER}" \
    --shard-size 20000 \
    --num-workers 128

echo ""
echo "=========================================="
echo "Kathbath done!"
echo "=========================================="
