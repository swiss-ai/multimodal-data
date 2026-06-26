#!/bin/bash
# Convert UPS WDS tars to Lhotse SHAR for a single language.
# Uses per-language VAD files produced by split_vad_by_lang.py.
#
# Submit with TARGET_LANG env var and appropriate --array range:
#   sbatch --export=ALL,TARGET_LANG=en --array=0-18 prepare_vad_to_shar_per_lang.slurm
#   sbatch --export=ALL,TARGET_LANG=af --array=0    prepare_vad_to_shar_per_lang.slurm
# Or use: bash submit_per_lang.sh en af
# Canary without touching the default output tree:
#   SHAR_ROOT_OVERRIDE=/some/other/output/en bash submit_per_lang.sh en
#
#SBATCH --account=infra01
#SBATCH --job-name=ups-lang-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/ups_lang_shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/convert/ups_lang_shar_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

if [ -z "${TARGET_LANG:-}" ]; then
  echo "ERROR: TARGET_LANG env var is required (e.g. TARGET_LANG=en)" >&2
  exit 1
fi

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
RUNTIME_ENV_SH="${REPO_DIR}/scripts/utils/source_lhotse_runtime.sh"
LOG_DIR="${REPO_DIR}/logs"

DATA_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/unsupervised_peoples_speech_commercial_wds
WDS_SHARDS_DIRS="${DATA_ROOT}/audio,${DATA_ROOT}/audio2"
VAD_PER_SHARD_DIR="${DATA_ROOT}/vad_per_lang/${TARGET_LANG}"

DEFAULT_SHAR_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1/ups_per_lang/${TARGET_LANG}
SHAR_ROOT="${SHAR_ROOT_OVERRIDE:-${DEFAULT_SHAR_ROOT}}"

NUM_WORKERS_PER_NODE=288
SHARD_SIZE=20000
TARGET_SR=24000
MIN_SR=16000
VAD_MAX_CHUNK_SEC=200
VAD_MIN_CHUNK_SEC=5
VAD_SAMPLE_RATE=16000
VAD_MAX_MERGE_GAP_SEC=1.0
VAD_MAX_DURATION_SEC=3600

RANK="${SLURM_ARRAY_TASK_ID}"
HOST="$(hostname -s)"

mkdir -p "${LOG_DIR}" "${SHAR_ROOT}"

if [ ! -d "${VAD_PER_SHARD_DIR}" ]; then
  echo "ERROR: VAD dir not found for lang=${TARGET_LANG}: ${VAD_PER_SHARD_DIR}" >&2
  exit 1
fi

echo "=========================================="
echo "UPS Per-Language VAD → SHAR: lang=${TARGET_LANG}"
echo "=========================================="
echo "Job: ${SLURM_ARRAY_JOB_ID}, Task: ${RANK}"
echo "Host: ${HOST}, CPUs: ${SLURM_CPUS_PER_TASK}"
echo "VAD dir: ${VAD_PER_SHARD_DIR}"
echo "Output: ${SHAR_ROOT}"
if [ -n "${SHAR_ROOT_OVERRIDE:-}" ]; then
  echo "Output override active via SHAR_ROOT_OVERRIDE"
fi
echo "=========================================="

# Validate WDS dirs
IFS=',' read -r -a SHARD_DIR_ARR <<< "${WDS_SHARDS_DIRS}"
EXISTING_WDS_DIRS=()
for d in "${SHARD_DIR_ARR[@]}"; do
  d_trimmed="$(echo "$d" | sed 's/^ *//;s/ *$//')"
  if [ -n "${d_trimmed}" ] && [ -d "${d_trimmed}" ]; then
    EXISTING_WDS_DIRS+=("${d_trimmed}")
  fi
done
if [ "${#EXISTING_WDS_DIRS[@]}" -eq 0 ]; then
  echo "ERROR: no WDS dirs found: ${WDS_SHARDS_DIRS}" >&2
  exit 1
fi

cd "${REPO_DIR}"
source "${RUNTIME_ENV_SH}"

# Discover tars from VAD files (fast: iterate small VAD dir, not all tars)
# VAD files are named {parent}_{stem}.jsonl -> tar is DATA_ROOT/{parent}/{stem}.tar
mapfile -t ALL_SHARDS < <(
  for vad_file in "${VAD_PER_SHARD_DIR}"/*.jsonl; do
    [ -f "${vad_file}" ] || continue
    base="$(basename "${vad_file}" .jsonl)"
    parent="${base%%_*}"
    stem="${base#*_}"
    tar_path="${DATA_ROOT}/${parent}/${stem}.tar"
    [ -f "${tar_path}" ] && echo "${tar_path}"
  done | sort -u
)
TOTAL="${#ALL_SHARDS[@]}"
if [ "${TOTAL}" -eq 0 ]; then
  echo "[rank ${RANK}] No tars found for lang=${TARGET_LANG}" >&2
  exit 0
fi

# Determine world size from array range
WORLD=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))

# Distribute shards across WORLD jobs
BASE=$((TOTAL / WORLD))
REM=$((TOTAL % WORLD))
if [ "${RANK}" -lt "$((SLURM_ARRAY_TASK_MIN + REM))" ]; then
  COUNT=$((BASE + 1))
  LOCAL_RANK=$((RANK - SLURM_ARRAY_TASK_MIN))
  START=$((LOCAL_RANK * COUNT))
else
  COUNT=${BASE}
  LOCAL_RANK=$((RANK - SLURM_ARRAY_TASK_MIN))
  START=$((REM * (BASE + 1) + (LOCAL_RANK - REM) * BASE))
fi

if [ "${COUNT}" -le 0 ]; then
  echo "[rank ${RANK}] No shards assigned (TOTAL=${TOTAL}, WORLD=${WORLD})"
  exit 0
fi

END=$((START + COUNT - 1))
ASSIGNED_TARS=("${ALL_SHARDS[@]:$START:$COUNT}")

NODE_TAG="$(printf "%02d" "${LOCAL_RANK}")"
NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"
mkdir -p "${NODE_SHAR_DIR}"

FIRST="$(basename "${ASSIGNED_TARS[0]}")"
LAST="$(basename "${ASSIGNED_TARS[-1]}")"

cat > "${NODE_SHAR_DIR}/node_assignment.txt" <<EOF
lang=${TARGET_LANG}
job_id=${SLURM_ARRAY_JOB_ID}
task_id=${RANK}
host=${HOST}
total_shards=${TOTAL}
assigned_count=${COUNT}
start_index=${START}
end_index=${END}
first_shard=${FIRST}
last_shard=${LAST}
world=${WORLD}
EOF

echo "[rank ${RANK}] lang=${TARGET_LANG} assigned=${COUNT}/${TOTAL} shards (${FIRST} .. ${LAST})"
echo "[rank ${RANK}] output=${NODE_SHAR_DIR}"

python -m audio_tokenization.prepare.prepare_wds_to_shar \
  --wds-shards "${ASSIGNED_TARS[@]}" \
  --shar-dir "${NODE_SHAR_DIR}" \
  --vad-segmentation \
  --vad-per-shard-dir "${VAD_PER_SHARD_DIR}" \
  --num-workers "${NUM_WORKERS_PER_NODE}" \
  --shard-size "${SHARD_SIZE}" \
  --target-sr "${TARGET_SR}" \
  --min-sr "${MIN_SR}" \
  --shar-format flac \
  --vad-max-chunk-sec "${VAD_MAX_CHUNK_SEC}" \
  --vad-min-chunk-sec "${VAD_MIN_CHUNK_SEC}" \
  --vad-sample-rate "${VAD_SAMPLE_RATE}" \
  --vad-max-merge-gap-sec "${VAD_MAX_MERGE_GAP_SEC}" \
  --vad-max-duration-sec "${VAD_MAX_DURATION_SEC}"

echo "[rank ${RANK}] lang=${TARGET_LANG} finished"
