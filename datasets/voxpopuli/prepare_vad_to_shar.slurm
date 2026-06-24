#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=vad2shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/voxpopuli_vad2shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/voxpopuli_vad2shar_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer

# Input dataset roots
AUDIO_ROOT=${AUDIO_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/voxpopuli/raw_audios}
VAD_DIR=${VAD_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/voxpopuli/vad_results_merged/per_lang_year_dedup}

# Shared output root (each job writes to SHAR_ROOT/node_XX)
SHAR_ROOT=${SHAR_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1_rebuild/voxpopuli_shar_v2}

# Processing parameters
NUM_WORKERS_PER_NODE=${NUM_WORKERS_PER_NODE:-288}
SHARD_SIZE=${SHARD_SIZE:-2000}
TARGET_SR=${TARGET_SR:-24000}
MIN_SR=${MIN_SR:-16000}
AUDIO_EXT=${AUDIO_EXT:-.ogg}
SHAR_FORMAT=${SHAR_FORMAT:-flac}
VAD_MAX_CHUNK_SEC=${VAD_MAX_CHUNK_SEC:-200}
VAD_MIN_CHUNK_SEC=${VAD_MIN_CHUNK_SEC:-5}
VAD_SAMPLE_RATE=${VAD_SAMPLE_RATE:-16000}
VAD_MAX_MERGE_GAP_SEC=${VAD_MAX_MERGE_GAP_SEC:-2.0}
VAD_MAX_DURATION_SEC=${VAD_MAX_DURATION_SEC:-200}

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD=4
HOST="$(hostname -s)"

chmod u+w "$(dirname "${SHAR_ROOT}")"
mkdir -p "${SHAR_ROOT}"
mkdir -p "${REPO_DIR}/logs"

# Discover all JSONL files and distribute across nodes
mapfile -t ALL_JSONLS < <(find "${VAD_DIR}" -name '*.jsonl' | sort)
TOTAL="${#ALL_JSONLS[@]}"

echo "=========================================="
echo "VoxPopuli VAD-Segmented Shar Prepare"
echo "=========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${RANK}/${WORLD}"
echo "Host: ${HOST}"
echo "Audio root: ${AUDIO_ROOT}"
echo "VAD dir: ${VAD_DIR}"
echo "JSONL files: ${TOTAL}"
echo "Output root: ${SHAR_ROOT}"
echo "Workers/node: ${NUM_WORKERS_PER_NODE}"
echo "VAD params: max_chunk=${VAD_MAX_CHUNK_SEC}s min_chunk=${VAD_MIN_CHUNK_SEC}s merge_gap=${VAD_MAX_MERGE_GAP_SEC}s max_dur=${VAD_MAX_DURATION_SEC}s"
echo "=========================================="

if [ ! -d "${AUDIO_ROOT}" ]; then
  echo "ERROR: Audio root not found: ${AUDIO_ROOT}" >&2
  exit 1
fi

if [ "${TOTAL}" -eq 0 ]; then
  echo "ERROR: No JSONL files found in ${VAD_DIR}" >&2
  exit 1
fi

# Distribute JSONL files across WORLD jobs (round-robin)
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
  echo "[rank ${RANK}] No JSONL files assigned (TOTAL=${TOTAL}, WORLD=${WORLD})"
  exit 0
fi

ASSIGNED_JSONLS=("${ALL_JSONLS[@]:$START:$COUNT}")

NODE_TAG="$(printf "%02d" "${RANK}")"
NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"
mkdir -p "${NODE_SHAR_DIR}"

echo "[rank ${RANK}] host=${HOST} assigned=${COUNT} JSONL files"
echo "[rank ${RANK}] output=${NODE_SHAR_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python -m audio_tokenization.prepare.prepare_audio_dir_to_shar \
  --audio-root "${AUDIO_ROOT}" \
  --jsonl-files "${ASSIGNED_JSONLS[@]}" \
  --shar-dir "${NODE_SHAR_DIR}" \
  --audio-ext "${AUDIO_EXT}" \
  --target-sr "${TARGET_SR}" \
  --shar-format "${SHAR_FORMAT}" \
  --shard-size "${SHARD_SIZE}" \
  --num-workers "${NUM_WORKERS_PER_NODE}" \
  --min-sr "${MIN_SR}" \
  --vad-max-chunk-sec "${VAD_MAX_CHUNK_SEC}" \
  --vad-min-chunk-sec "${VAD_MIN_CHUNK_SEC}" \
  --vad-sample-rate "${VAD_SAMPLE_RATE}" \
  --vad-max-merge-gap-sec "${VAD_MAX_MERGE_GAP_SEC}" \
  --vad-max-duration-sec "${VAD_MAX_DURATION_SEC}" \
  --resampling-backend torchaudio

echo "[rank ${RANK}] finished"
