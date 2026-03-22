#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=ps-dirty-to-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/peoples_speech/logs/ps_dirty_to_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/peoples_speech/logs/ps_dirty_to_shar_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

unset LD_PRELOAD
export MALLOC_CHECK_=0

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
REPO_DIR=${REPO_DIR:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

ARROW_DIR=${ARROW_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/people_speech_dirty_recap}
ARROW_GLOB=${ARROW_GLOB:-data-*.arrow}
SHAR_ROOT=${SHAR_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/peoples_speech_dirty_train}
TEXT_TOKENIZER=${TEXT_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json}

NUM_WORKERS_PER_NODE=${NUM_WORKERS_PER_NODE:-288}
SHARD_SIZE=${SHARD_SIZE:-5000}
TARGET_SR=${TARGET_SR:-24000}
SHAR_FORMAT=${SHAR_FORMAT:-flac}
RESAMPLING_BACKEND=${RESAMPLING_BACKEND:-sox}

RANK="${SLURM_ARRAY_TASK_ID:-0}"
WORLD="${WORLD:-${SLURM_ARRAY_TASK_COUNT:-1}}"
HOST="$(hostname -s)"

echo "=========================================="
echo "People's Speech Arrow → Shar"
echo "=========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${RANK}/${WORLD}"
echo "Host: ${HOST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Arrow dir: ${ARROW_DIR}"
echo "Arrow glob: ${ARROW_GLOB}"
echo "Output root: ${SHAR_ROOT}"
echo "Workers/node: ${NUM_WORKERS_PER_NODE}"
echo "=========================================="

# Discover all train arrow files
mapfile -t ALL_ARROWS < <(
  find "${ARROW_DIR}" -maxdepth 1 -type f -name "${ARROW_GLOB}" | sort
)
TOTAL="${#ALL_ARROWS[@]}"

if [ "${TOTAL}" -eq 0 ]; then
  echo "[rank ${RANK}] ERROR: no arrow files found" >&2
  exit 1
fi

# Distribute arrow files across WORLD jobs (contiguous chunks)
BASE=$((TOTAL / WORLD))
REM=$((TOTAL % WORLD))
if [ "${RANK}" -lt "${REM}" ]; then
  COUNT=$((BASE + 1))
  START=$((RANK * COUNT))
else
  COUNT=${BASE}
  START=$((REM * (BASE + 1) + (RANK - REM) * BASE))
fi

if [ "${COUNT}" -le 0 ]; then
  echo "[rank ${RANK}] No arrows assigned (TOTAL=${TOTAL}, WORLD=${WORLD})"
  exit 0
fi

ASSIGNED_ARROWS=("${ALL_ARROWS[@]:$START:$COUNT}")

if [ "${WORLD}" -eq 1 ]; then
  NODE_SHAR_DIR="${SHAR_ROOT}"
else
  NODE_TAG="$(printf "%02d" "${RANK}")"
  NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"
fi
mkdir -p "${NODE_SHAR_DIR}"

echo "[rank ${RANK}] host=${HOST} assigned=${COUNT} arrow files"
echo "[rank ${RANK}] output=${NODE_SHAR_DIR}"

cd "${REPO_DIR}"

"${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_hf_to_shar \
  --arrow-files "${ASSIGNED_ARROWS[@]}" \
  --shar-dir "${NODE_SHAR_DIR}" \
  --audio-column audio \
  --text-column text \
  --id-column id \
  --target-sr "${TARGET_SR}" \
  --shard-size "${SHARD_SIZE}" \
  --shar-format "${SHAR_FORMAT}" \
  --text-tokenizer "${TEXT_TOKENIZER}" \
  --num-workers "${NUM_WORKERS_PER_NODE}" \
  --resampling-backend "${RESAMPLING_BACKEND}"

echo "[rank ${RANK}] finished"
