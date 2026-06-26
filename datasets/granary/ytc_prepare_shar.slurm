#!/bin/bash
# Granary YTC English → Shar (464GB, 1.3M wav segments, NVIDIA Whisper pseudo-labels)
# Audio: /capstor/.../granary/ytc_tars/en/*.tar (wav files, no text)
# Labels: /capstor/.../raw/granary/nvidia-granary/en/ytc/en_asr.jsonl
#SBATCH --account=infra01
#SBATCH --job-name=granary-ytc
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/granary-ytc-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/granary-ytc-%x-%A.err
#SBATCH --time=12:00:00
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
export FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
export PATH=${FFMPEG_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}

SHAR_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/granary_ytc_en
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

echo "=========================================="
echo "Granary YTC English → Shar"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output: ${SHAR_DIR}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
    --wds-shards '/capstor/store/cscs/swissai/infra01/audio-datasets/granary/ytc_tars/en/*.tar' \
    --external-metadata /capstor/store/cscs/swissai/infra01/audio-datasets/raw/granary/nvidia-granary/en/ytc/en_asr.jsonl \
    --id-field utt_id \
    --text-field text \
    --custom-fields source_lang dataset_source offset \
    --shar-dir "${SHAR_DIR}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --text-tokenizer "${TOKENIZER}" \
    --num-workers 128

echo ""
echo "=========================================="
echo "Granary YTC English done!"
echo "=========================================="
