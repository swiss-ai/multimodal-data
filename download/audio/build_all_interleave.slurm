#!/bin/bash
# Build final interleaved indexed datasets from all interleave_cache datasets.
# Shift-by-one mode: always ends with text, two offsets + transcribe.
# Routes to stage2/ (≤8192 tokens) and lct/ (>8192 tokens).
#
# Usage:
#   sbatch --array=0-13 build_all_interleave.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=build-intlv
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/build/interleave_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/build/interleave_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
uv pip install --python /opt/venv/bin/python orjson polars

CACHE_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/interleave_cache
OUT_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/tokenized/interleave
TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/build

DATASETS=(
    aishell4_interleave
    eurospeech_interleave
    granary_yodas_interleave
    legco_speech_interleave
    libriheavy_large_interleave
    mls_7lang_interleave
    parlaspeech_rs_interleave
    peoples_speech_train_interleaved
    spc_r_interleave
    vimedcss_interleave
    voxpopuli_asr_interleave
    wenetspeech_interleave
    zeroth_korean
    zoengjyutgaai_interleave
)

# Strip _interleave/_interleaved suffix for output name
OUTPUT_NAMES=(
    aishell4
    eurospeech
    granary_yodas
    legco_speech
    libriheavy_large
    mls_7lang
    parlaspeech_rs
    peoples_speech
    spc_r
    vimedcss
    voxpopuli_asr
    wenetspeech
    zeroth_korean
    zoengjyutgaai
)

ds="${DATASETS[${SLURM_ARRAY_TASK_ID}]}"
out="${OUTPUT_NAMES[${SLURM_ARRAY_TASK_ID}]}"

echo "=========================================="
echo "Build interleave: ${ds} → ${out}"
echo "Task ${SLURM_ARRAY_TASK_ID}, Job ${SLURM_JOB_ID}"
echo "=========================================="

python -m audio_tokenization.utils.build_interleaved.shift_by_one \
    --parquet-dir "${CACHE_BASE}/${ds}" \
    --output-dir "${OUT_BASE}/${out}" \
    --tokenizer-path "${TOKENIZER}" \
    --max-seq-len 262144 \
    --seq-threshold 8192 \
    --transcribe-ratio 0.4 \
    --tmp-dir /tmp \
    --num-workers 256

echo "Done: ${out}"
