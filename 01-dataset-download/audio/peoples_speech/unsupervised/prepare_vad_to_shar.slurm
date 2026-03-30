#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=vad2shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/ups_all_vad2shar_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer/logs/ups_all_vad2shar_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --array=0-26
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

REPO_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
LOG_DIR="${REPO_DIR}/logs"

# Input dataset roots
DATA_ROOT=/iopsstor/scratch/cscs/xyixuan/audio-datasets/raw/unsupervised_peoples_speech_commercial_wds
WDS_SHARDS_DIRS=${WDS_SHARDS_DIRS:-${DATA_ROOT}/audio,${DATA_ROOT}/audio2}

# Per-shard VAD JSONL directory (produced by filter_langid_vad.py --languages all)
VAD_PER_SHARD_DIR=${VAD_PER_SHARD_DIR:-${DATA_ROOT}/vad_results_all_per_shard}

# Shared output root (each job writes to SHAR_ROOT/node_XX)
SHAR_ROOT=${SHAR_ROOT:-/iopsstor/scratch/cscs/xyixuan/audio-datasets/shar/ups_all_lang_shar_v2}

# Per-node tuning
NUM_WORKERS_PER_NODE=${NUM_WORKERS_PER_NODE:-288}
SHARD_SIZE=${SHARD_SIZE:-5000}
TARGET_SR=${TARGET_SR:-24000}
MIN_SR=${MIN_SR:-16000}
VAD_MAX_CHUNK_SEC=${VAD_MAX_CHUNK_SEC:-200}
VAD_MIN_CHUNK_SEC=${VAD_MIN_CHUNK_SEC:-5}
VAD_SAMPLE_RATE=${VAD_SAMPLE_RATE:-16000}
VAD_MAX_MERGE_GAP_SEC=${VAD_MAX_MERGE_GAP_SEC:-2.0}
VAD_MAX_DURATION_SEC=${VAD_MAX_DURATION_SEC:-200}

RANK="${SLURM_ARRAY_TASK_ID}"
WORLD=27
HOST="$(hostname -s)"

mkdir -p "${LOG_DIR}"
mkdir -p "${SHAR_ROOT}"

echo "=========================================="
echo "UPS All-Language VAD-Segmented Shar Prepare"
echo "=========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}, Task: ${RANK}/${WORLD}"
echo "Host: ${HOST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "WDS dirs: ${WDS_SHARDS_DIRS}"
echo "VAD per-shard dir: ${VAD_PER_SHARD_DIR}"
echo "Output root: ${SHAR_ROOT}"
echo "Workers/node: ${NUM_WORKERS_PER_NODE}"
echo "VAD params: max_chunk=${VAD_MAX_CHUNK_SEC}s min_chunk=${VAD_MIN_CHUNK_SEC}s merge_gap=${VAD_MAX_MERGE_GAP_SEC}s max_dur=${VAD_MAX_DURATION_SEC}s"
echo "=========================================="

IFS=',' read -r -a SHARD_DIR_ARR <<< "${WDS_SHARDS_DIRS}"
EXISTING_WDS_DIRS=()
for d in "${SHARD_DIR_ARR[@]}"; do
  d_trimmed="$(echo "$d" | sed 's/^ *//;s/ *$//')"
  if [ -n "${d_trimmed}" ] && [ -d "${d_trimmed}" ]; then
    EXISTING_WDS_DIRS+=("${d_trimmed}")
  fi
done

if [ "${#EXISTING_WDS_DIRS[@]}" -eq 0 ]; then
  echo "ERROR: none of WDS shard directories exist: ${WDS_SHARDS_DIRS}" >&2
  exit 1
fi

if [ ! -d "${VAD_PER_SHARD_DIR}" ]; then
  echo "ERROR: VAD per-shard directory not found: ${VAD_PER_SHARD_DIR}" >&2
  exit 1
fi

cd "${REPO_DIR}"
uv pip install --no-deps --no-build-isolation git+https://github.com/pytorch/audio.git@release/2.9
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Discover all tars that have a corresponding VAD file
mapfile -t ALL_SHARDS < <(
  for d in "${EXISTING_WDS_DIRS[@]}"; do
    for tar in "${d}"/*.tar; do
      [ -f "${tar}" ] || continue
      parent="$(basename "$(dirname "${tar}")")"
      stem="$(basename "${tar}" .tar)"
      vad_file="${VAD_PER_SHARD_DIR}/${parent}_${stem}.jsonl"
      [ -f "${vad_file}" ] && echo "${tar}"
    done
  done | sort
)
TOTAL="${#ALL_SHARDS[@]}"
if [ "${TOTAL}" -eq 0 ]; then
  echo "[rank ${RANK}] ERROR: no tars with VAD files found under ${WDS_SHARDS_DIRS}" >&2
  exit 1
fi

# Distribute shards across WORLD jobs
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
  echo "[rank ${RANK}] No shards assigned (TOTAL=${TOTAL}, WORLD=${WORLD})"
  exit 0
fi

END=$((START + COUNT - 1))
FIRST="$(basename "${ALL_SHARDS[${START}]}")"
LAST="$(basename "${ALL_SHARDS[${END}]}")"

ASSIGNED_TARS=("${ALL_SHARDS[@]:$START:$COUNT}")

NODE_TAG="$(printf "%02d" "${RANK}")"
NODE_SHAR_DIR="${SHAR_ROOT}/node_${NODE_TAG}"
mkdir -p "${NODE_SHAR_DIR}"

cat > "${NODE_SHAR_DIR}/node_assignment.txt" <<EOF
job_id=${SLURM_ARRAY_JOB_ID}
task_id=${RANK}
host=${HOST}
total_shards=${TOTAL}
assigned_count=${COUNT}
start_index=${START}
end_index=${END}
first_shard=${FIRST}
last_shard=${LAST}
workers_per_node=${NUM_WORKERS_PER_NODE}
EOF

echo "[rank ${RANK}] host=${HOST} assigned=${COUNT} shards (${FIRST} .. ${LAST})"
echo "[rank ${RANK}] output=${NODE_SHAR_DIR}"

python -m audio_tokenization.utils.prepare_data.prepare_wds_to_shar \
  --wds-shards "${ASSIGNED_TARS[@]}" \
  --shar-dir "${NODE_SHAR_DIR}" \
  --vad-segmentation \
  --vad-per-shard-dir "${VAD_PER_SHARD_DIR}" \
  --num-workers "${NUM_WORKERS_PER_NODE}" \
  --shard-size "${SHARD_SIZE}" \
  --target-sr "${TARGET_SR}" \
  --min-sr "${MIN_SR}" \
  --text-ext "" \
  --shar-format flac \
  --vad-max-chunk-sec "${VAD_MAX_CHUNK_SEC}" \
  --vad-min-chunk-sec "${VAD_MIN_CHUNK_SEC}" \
  --vad-sample-rate "${VAD_SAMPLE_RATE}" \
  --vad-max-merge-gap-sec "${VAD_MAX_MERGE_GAP_SEC}" \
  --vad-max-duration-sec "${VAD_MAX_DURATION_SEC}"

echo "[rank ${RANK}] finished"
