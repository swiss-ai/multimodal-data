#!/bin/bash
# Convert CoRal v3 parquet splits to SHAR.
# By default, converts both subsets (read_aloud + conversation) for train/val/test.
#
# Example:
#   sbatch prepare_coral_v3_to_shar.slurm
#
# Override at submit time:
#   sbatch --export=ALL,SUBSETS="read_aloud",SPLITS="train",NUM_WORKERS=128 prepare_coral_v3_to_shar.slurm

#SBATCH --account=infra01
#SBATCH --job-name=coral-v3-to-shar
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/coral-v3-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/coral-v3-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=PA-2338-RL

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
LHOTSE_REPO=${LHOTSE_REPO:-/iopsstor/scratch/cscs/xyixuan/dev/lhotse}
BENCHMARK_REPO=${BENCHMARK_REPO:-/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer}

RAW_ROOT=${RAW_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/coral_v3}
SHAR_STAGE2=${SHAR_STAGE2:-/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2}
SHAR_SUFFIX=${SHAR_SUFFIX:-}

FFMPEG_ROOT=${FFMPEG_ROOT:-/capstor/store/cscs/swissai/infra01/MLLM/wheelhouse/aarch64/ffmpeg-7.1.1-full-aarch64}
TEXT_TOKENIZER=${TEXT_TOKENIZER:-/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json}

SUBSETS=${SUBSETS:-read_aloud conversation}
SPLITS=${SPLITS:-train}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-288}}
TARGET_SR=${TARGET_SR:-24000}
SHARD_SIZE=${SHARD_SIZE:-5000}
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-1}

export PATH="${FFMPEG_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LHOTSE_REPO}:${BENCHMARK_REPO}:${PYTHONPATH:-}"
uv pip install --python /opt/venv/bin/python polars

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "ERROR: RAW_ROOT does not exist: ${RAW_ROOT}" >&2
  exit 1
fi

id_column_for_subset() {
  local subset="$1"
  case "${subset}" in
    read_aloud) echo "id_recording" ;;
    conversation) echo "id_conversation" ;;
    *)
      echo "ERROR: unsupported subset '${subset}' (supported: read_aloud, conversation)" >&2
      return 1
      ;;
  esac
}

clip_id_parser_for_subset() {
  local subset="$1"
  case "${subset}" in
    read_aloud) echo "" ;;
    conversation) echo "trailing_number" ;;
    *)
      echo "ERROR: unsupported subset '${subset}' (supported: read_aloud, conversation)" >&2
      return 1
      ;;
  esac
}

echo "=========================================="
echo "CoRal v3 -> SHAR"
echo "Raw root:        ${RAW_ROOT}"
echo "SHAR stage_2:    ${SHAR_STAGE2}"
echo "SHAR suffix:     ${SHAR_SUFFIX:-<none>}"
echo "Subsets:         ${SUBSETS}"
echo "Splits:          ${SPLITS}"
echo "Workers:         ${NUM_WORKERS}"
echo "Target SR:       ${TARGET_SR}"
echo "Shard size:      ${SHARD_SIZE}"
echo "Skip existing:   ${SKIP_IF_EXISTS}"
echo "Python:          ${PYTHON_BIN}"
echo "FFmpeg root:     ${FFMPEG_ROOT}"
echo "=========================================="

  for subset in ${SUBSETS}; do
    parquet_dir="${RAW_ROOT}/${subset}"
    id_column="$(id_column_for_subset "${subset}")"
    input_clip_id_parser="$(clip_id_parser_for_subset "${subset}")"

  if [[ ! -d "${parquet_dir}" ]]; then
    echo "[${subset}] WARNING: missing subset directory, skipping: ${parquet_dir}"
    continue
  fi

  for split in ${SPLITS}; do
    parquet_glob="${split}-*.parquet"
    shar_dir="${SHAR_STAGE2}/coral_v3_${subset}_${split}${SHAR_SUFFIX}"

    if ! compgen -G "${parquet_dir}/${parquet_glob}" > /dev/null; then
      echo "[${subset}/${split}] WARNING: no parquet matched ${parquet_glob}, skipping"
      continue
    fi

    if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${shar_dir}/shar_index.json" ]]; then
      echo "[${subset}/${split}] Found existing shar_index.json, skipping: ${shar_dir}"
      continue
    fi

    prepare_args=(
      --parquet-dir "${parquet_dir}"
      --parquet-glob "${parquet_glob}"
      --shar-dir "${shar_dir}"
      --id-column "${id_column}"
      --text-column text
      --target-sr "${TARGET_SR}"
      --text-tokenizer "${TEXT_TOKENIZER}"
      --shard-size "${SHARD_SIZE}"
      --num-workers "${NUM_WORKERS}"
    )
    if [[ -n "${input_clip_id_parser}" ]]; then
      prepare_args+=(--input-clip-id-parser "${input_clip_id_parser}")
    fi

    echo "[${subset}/${split}] START -> ${shar_dir}"
    "${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_parquet_to_shar \
      "${prepare_args[@]}"
    echo "[${subset}/${split}] DONE"
  done
done

echo "All requested CoRal v3 conversions finished."
