#!/bin/bash
# SeamlessAlign → Shar (5 Indian languages, transcribe + translate)
# Two passes per language: stage_2 (source text) + stage_2/translate (English)
#SBATCH --account=infra01
#SBATCH --job-name=seamless-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/seamless-shar-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/convert/seamless-shar-%x-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --reservation=PA-2338-RL
#SBATCH --exclusive

set -euo pipefail

ulimit -n 65536

uv pip install --python /opt/venv/bin/python polars

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}
export FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
export PATH=${FFMPEG_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}

RAW_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___ai4bharat___SeamlessAlign/Indic-En
SHAR_TRANSCRIBE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/seamless_align
SHAR_TRANSLATE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/translate/seamless_align
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

LANGUAGES=(hin kan tam tel urd)
LANG="${LANGUAGES[${SLURM_ARRAY_TASK_ID}]}"

echo "=========================================="
echo "SeamlessAlign → Shar (${LANG})"
echo "  Transcribe + Translate"
echo "=========================================="

# Pass 1: Transcribe (source language text)
echo "--- ${LANG} transcribe ---"
python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_ROOT}/${LANG}" \
    --parquet-glob '*.parquet' \
    --shar-dir "${SHAR_TRANSCRIBE}/${LANG}_train" \
    --audio-column audio_filepath \
    --text-column transcription \
    --duration-column duration \
    --custom-columns sonar_score laser_score text_lid_score \
    --target-sr 24000 \
    --text-tokenizer "${TOKENIZER}" \
    --shard-size 20000 \
    --num-workers 32

# Pass 2: Translate (English text)
echo "--- ${LANG} translate ---"
python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_ROOT}/${LANG}" \
    --parquet-glob '*.parquet' \
    --shar-dir "${SHAR_TRANSLATE}/${LANG}_train" \
    --audio-column audio_filepath \
    --text-column text \
    --duration-column duration \
    --custom-columns sonar_score laser_score text_lid_score \
    --target-sr 24000 \
    --text-tokenizer "${TOKENIZER}" \
    --shard-size 20000 \
    --num-workers 32

echo ""
echo "=========================================="
echo "SeamlessAlign ${LANG} done!"
echo "=========================================="
