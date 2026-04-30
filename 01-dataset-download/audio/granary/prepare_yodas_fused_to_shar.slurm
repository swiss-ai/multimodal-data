#!/bin/bash
# Convert YODAS ASR+AST fused to SHAR.
# Reads both asr_only/ and ast/ parquets per shard, uses original language text.
# translation_en stored in custom for later translate tokenization.
# One array task per shard (92 shards). Post-hoc patch_universal_ids needed after.
#
# Usage:
#   sbatch --array=0-49 prepare_yodas_fused_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=yodas-fused
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/yodas_fused_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/yodas_fused_shar_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

DATA_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/granary/yodas-granary/data
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/granary_yodas_interleave

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson polars zstandard

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
mapfile -t ALL_SHARDS < <(ls "${DATA_ROOT}" | sort)
TOTAL="${#ALL_SHARDS[@]}"

echo "=========================================="
echo "YODAS fused → SHAR: rank ${RANK}/${WORLD}, ${TOTAL} total shards"
echo "=========================================="

for (( i=RANK; i<TOTAL; i+=WORLD )); do
  SHARD="${ALL_SHARDS[$i]}"
  SHAR_DIR="${SHAR_BASE}/${SHARD}"

  if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
    echo "[rank ${RANK}] ${SHARD} already done, skipping"
    continue
  fi

  MY_DIR="/tmp/yodas_fused_${RANK}_${SLURM_JOB_ID}"
  rm -rf "${MY_DIR}"
  mkdir -p "${MY_DIR}"

  for split in asr_only ast; do
    if [ -d "${DATA_ROOT}/${SHARD}/${split}" ]; then
      for pq in "${DATA_ROOT}/${SHARD}/${split}"/*.parquet; do
        [ -f "${pq}" ] || continue
        ln -s "${pq}" "${MY_DIR}/${split}_$(basename ${pq})"
      done
    fi
  done

  NUM_PQ=$(ls "${MY_DIR}"/*.parquet 2>/dev/null | wc -l)
  if [ "${NUM_PQ}" -eq 0 ]; then
    echo "[rank ${RANK}] ${SHARD}: no parquets, skipping"
    rm -rf "${MY_DIR}"
    continue
  fi

  NUM_WORKERS=32
  if [ "${NUM_PQ}" -lt "${NUM_WORKERS}" ]; then
    NUM_WORKERS="${NUM_PQ}"
  fi

  echo "[rank ${RANK}] ${SHARD}: ${NUM_PQ} parquets, ${NUM_WORKERS} workers"

  python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
    --parquet-dir "${MY_DIR}" \
    --parquet-glob "*.parquet" \
    --shar-dir "${SHAR_DIR}" \
    --id-column utt_id \
    --text-column text \
    --audio-column audio \
    --language-column lang \
    --custom-columns translation_en task original_audio_id original_audio_offset \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --target-sr 24000 \
    --shard-size 20000 \
    --num-workers "${NUM_WORKERS}"

  rm -rf "${MY_DIR}"
  echo "[rank ${RANK}] ${SHARD} done"
done

echo "[rank ${RANK}] all done"
