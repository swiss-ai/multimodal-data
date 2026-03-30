#!/bin/bash
# EuroSpeech permissive languages → Lhotse Shar (one language per array task)
# Italy (CC BY 4.0), UK (OGL v3.0), Bulgaria (CC BY 2.5), France (Licence Ouverte), Norway (NLOD 2.0)
#SBATCH --account=infra01
#SBATCH --job-name=eurospeech-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/eurospeech-shar-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/eurospeech-shar-%x-%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL
#SBATCH --partition=normal
#SBATCH --exclusive
#SBATCH --array=0-4

set -euo pipefail

LANGUAGES=(italy uk bulgaria france norway)
LANG=${LANGUAGES[$SLURM_ARRAY_TASK_ID]}

RAW_DIR=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___disco-eth___EuroSpeech
SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/eurospeech
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json
NUM_WORKERS=128

ulimit -n 65536

uv pip install --python /opt/venv/bin/python polars

export PYTHONPATH=/iopsstor/scratch/cscs/xyixuan/dev/lhotse:/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer:${PYTHONPATH:-}
export FFMPEG_ROOT=/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64
export PATH=${FFMPEG_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "EuroSpeech → Shar: ${LANG}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}, Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Input: ${RAW_DIR}/${LANG}/train-*.parquet"
echo "Output: ${SHAR_ROOT}/${LANG}_train"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${RAW_DIR}/${LANG}" \
    --parquet-glob 'train-*.parquet' \
    --shar-dir "${SHAR_ROOT}/${LANG}_train" \
    --audio-column audio \
    --text-column human_transcript \
    --id-column key \
    --language-column language \
    --duration-column duration_seconds \
    --target-sr 24000 \
    --text-tokenizer "${TOKENIZER}" \
    --shard-size 5000 \
    --num-workers ${NUM_WORKERS}

echo ""
echo "=========================================="
echo "${LANG} done!"
echo "=========================================="
