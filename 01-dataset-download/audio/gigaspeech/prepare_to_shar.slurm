#!/bin/bash
# Convert GigaSpeech XL to SHAR (stage_2).
# 258 train parquets (~4GB each, WAV embedded), ~8.3M segments, ~10K hours.
# Each array task gets ~65 parquets, processed with 32 workers.
# Uses normalized text from external_metadata.jsonl.gz.
#
# Usage:
#   sbatch --array=0-3 prepare_to_shar.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=giga-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/gigaspeech_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/gigaspeech_shar_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/source_lhotse_runtime.sh"
TEXT_TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json

DATA_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/hf___speechcolab___gigaspeech
PARQUET_DIR="${DATA_ROOT}/parquet-data/xl"
METADATA="${DATA_ROOT}/external_metadata.jsonl.gz"
SHAR_BASE=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/gigaspeech

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"
uv pip install --python /opt/venv/bin/python orjson polars zstandard

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

# Distribute parquets round-robin across ranks
mapfile -t ALL_PARQUETS < <(ls "${PARQUET_DIR}"/train-*.parquet | sort)
TOTAL="${#ALL_PARQUETS[@]}"

# Create temp dir with symlinks to this rank's parquets
MY_DIR="/tmp/gigaspeech_rank_${RANK}_${SLURM_JOB_ID}"
rm -rf "${MY_DIR}"
mkdir -p "${MY_DIR}"

MY_COUNT=0
for (( i=RANK; i<TOTAL; i+=WORLD )); do
  ln -s "${ALL_PARQUETS[$i]}" "${MY_DIR}/"
  MY_COUNT=$((MY_COUNT + 1))
done

SHAR_DIR="${SHAR_BASE}/node_$(printf '%02d' ${RANK})"

if [ -f "${SHAR_DIR}/_SUCCESS" ]; then
  echo "[rank ${RANK}] already done, skipping"
  rm -rf "${MY_DIR}"
  exit 0
fi

echo "=========================================="
echo "GigaSpeech → SHAR: rank ${RANK}/${WORLD}, ${MY_COUNT} parquets"
echo "=========================================="

python -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
  --parquet-dir "${MY_DIR}" \
  --parquet-glob "*.parquet" \
  --external-metadata "${METADATA}" \
  --id-column segment_id \
  --id-field segment_id \
  --text-field text_norm \
  --audio-column audio \
  --custom-columns audio_id begin_time end_time \
  --shar-dir "${SHAR_DIR}" \
  --target-sr 24000 \
  --shard-size 20000 \
  --text-tokenizer "${TEXT_TOKENIZER}" \
  --language en \
  --num-workers "${MY_COUNT}"

rm -rf "${MY_DIR}"

echo "[rank ${RANK}] done"
