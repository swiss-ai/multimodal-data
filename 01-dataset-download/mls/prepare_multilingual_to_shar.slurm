#!/bin/bash
# Convert MLS Arrow cache to SHAR (multilingual + English).
#
# Default behavior:
# - 8 languages via array task id (0-7): english + 7 non-English MLS languages
# - train split only
#
# Examples:
#   sbatch prepare_multilingual_to_shar.slurm
#   sbatch --export=ALL,SPLITS="train dev test 1_hours 9_hours",NUM_WORKERS=128 prepare_multilingual_to_shar.slurm
#   sbatch --array=0-2 --export=ALL,LANGUAGES="german french spanish" prepare_multilingual_to_shar.slurm
#
# Notes:
# - Non-English cache root:
#   /capstor/store/cscs/swissai/infra01/audio-datasets/mls_cache/facebook___multilingual_librispeech
# - English cache root:
#   /capstor/store/cscs/swissai/infra01/audio-datasets/mls_cache/english/default/0.0.0
# - Column mapping is fixed:
#   --audio-column audio --text-column transcript
#   --id-column id (non-English), --id-column original_path (English)

#SBATCH --account=infra01
#SBATCH --job-name=mls-to-shar
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mls-%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/mls-%x-%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --array=0-7
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
BENCHMARK_REPO=${BENCHMARK_REPO:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}

MLS_MULTI_ROOT=${MLS_MULTI_ROOT:-${MLS_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/mls_cache/facebook___multilingual_librispeech}}
MLS_ENGLISH_ROOT=${MLS_ENGLISH_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/mls_cache/english/default/0.0.0}
SHAR_STAGE2=${SHAR_STAGE2:-/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2}
TEXT_TOKENIZER=${TEXT_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json}

LANGUAGES=${LANGUAGES:-"english dutch french german italian polish portuguese spanish"}
SPLITS=${SPLITS:-"train"}
TARGET_SR=${TARGET_SR:-24000}
SHARD_SIZE=${SHARD_SIZE:-5000}
SHAR_FORMAT=${SHAR_FORMAT:-flac}
RESAMPLING_BACKEND=${RESAMPLING_BACKEND:-sox}
NUM_WORKERS=${NUM_WORKERS:-64}
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-1}

export PYTHONPATH="${BENCHMARK_REPO}:${PYTHONPATH:-}"

read -r -a LANG_ARR <<< "${LANGUAGES}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if [[ "${TASK_ID}" -ge "${#LANG_ARR[@]}" ]]; then
  echo "[task ${TASK_ID}] No language mapped (LANGUAGES count=${#LANG_ARR[@]}). Exiting."
  exit 0
fi

LANGUAGE="${LANG_ARR[${TASK_ID}]}"
if [[ "${LANGUAGE}" == "english" ]]; then
  LANG_DIR="${MLS_ENGLISH_ROOT}"
else
  LANG_DIR="${MLS_MULTI_ROOT}/${LANGUAGE}/0.0.0"
fi

if [[ ! -d "${LANG_DIR}" ]]; then
  echo "ERROR: language cache directory not found: ${LANG_DIR}" >&2
  exit 1
fi

HASH_DIR="$(find "${LANG_DIR}" -mindepth 1 -maxdepth 1 -type d | sort | head -n1)"
if [[ -z "${HASH_DIR}" || ! -d "${HASH_DIR}" ]]; then
  echo "ERROR: no hash directory found under ${LANG_DIR}" >&2
  exit 1
fi

arrow_glob_for_split() {
  local language="$1"
  local split="$2"
  if [[ "${language}" == "english" ]]; then
    case "${split}" in
      train) echo "mls_eng-train-*.arrow" ;;
      dev) echo "mls_eng-dev.arrow" ;;
      test) echo "mls_eng-test.arrow" ;;
      *)
        echo "ERROR: unsupported split '${split}' for english (supported: train/dev/test)" >&2
        return 1
        ;;
    esac
  else
    case "${split}" in
      train) echo "multilingual_librispeech-train-*.arrow" ;;
      dev) echo "multilingual_librispeech-dev.arrow" ;;
      test) echo "multilingual_librispeech-test.arrow" ;;
      1_hours) echo "multilingual_librispeech-1_hours.arrow" ;;
      9_hours) echo "multilingual_librispeech-9_hours.arrow" ;;
      *)
        echo "ERROR: unsupported split '${split}'" >&2
        return 1
        ;;
    esac
  fi
}

id_column_for_language() {
  local language="$1"
  if [[ "${language}" == "english" ]]; then
    # English cache does not provide top-level "id"; use original_path.
    echo "original_path"
  else
    echo "id"
  fi
}

echo "=========================================="
echo "MLS -> SHAR"
echo "Task ID:         ${TASK_ID}"
echo "Language:        ${LANGUAGE}"
echo "MLS multi root:  ${MLS_MULTI_ROOT}"
echo "MLS eng root:    ${MLS_ENGLISH_ROOT}"
echo "Hash dir:        ${HASH_DIR}"
echo "SHAR stage_2:    ${SHAR_STAGE2}"
echo "Splits:          ${SPLITS}"
echo "Target SR:       ${TARGET_SR}"
echo "Shard size:      ${SHARD_SIZE}"
echo "Format:          ${SHAR_FORMAT}"
echo "Requested worker:${NUM_WORKERS}"
echo "Python:          ${PYTHON_BIN}"
echo "=========================================="

cd "${BENCHMARK_REPO}"

for split in ${SPLITS}; do
  arrow_glob="$(arrow_glob_for_split "${LANGUAGE}" "${split}")"
  id_column="$(id_column_for_language "${LANGUAGE}")"
  shar_dir="${SHAR_STAGE2}/mls_${LANGUAGE}_${split}"

  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${shar_dir}/shar_index.json" ]]; then
    echo "[${LANGUAGE}/${split}] shar_index.json exists, skipping: ${shar_dir}"
    continue
  fi

  mapfile -t ARROWS < <(find "${HASH_DIR}" -maxdepth 1 -type f -name "${arrow_glob}" | sort)
  if [[ "${#ARROWS[@]}" -eq 0 ]]; then
    echo "[${LANGUAGE}/${split}] WARNING: no arrow files matched ${arrow_glob}, skipping"
    continue
  fi

  workers="${NUM_WORKERS}"
  if [[ "${workers}" -gt "${#ARROWS[@]}" ]]; then
    workers="${#ARROWS[@]}"
  fi
  if [[ "${workers}" -lt 1 ]]; then
    workers=1
  fi

  echo "[${LANGUAGE}/${split}] START -> ${shar_dir}"
  echo "[${LANGUAGE}/${split}] arrows=${#ARROWS[@]} workers=${workers} id_column=${id_column}"

  "${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_hf_to_shar \
    --arrow-files "${ARROWS[@]}" \
    --shar-dir "${shar_dir}" \
    --audio-column audio \
    --text-column transcript \
    --id-column "${id_column}" \
    --target-sr "${TARGET_SR}" \
    --shard-size "${SHARD_SIZE}" \
    --shar-format "${SHAR_FORMAT}" \
    --text-tokenizer "${TEXT_TOKENIZER}" \
    --num-workers "${workers}" \
    --resampling-backend "${RESAMPLING_BACKEND}"

  echo "[${LANGUAGE}/${split}] DONE"
done

echo "[task ${TASK_ID}] Finished language ${LANGUAGE}"
