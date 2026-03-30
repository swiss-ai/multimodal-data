#!/usr/bin/env bash
# End-to-end CommonVoice pipeline for one language:
# 1) Download with conda env "tools"
# 2) Convert train split to SHAR/stage_2/commonvoice/<lang>_train
# 3) Convert other split to SHAR/stage_1/commonvoice/<lang>_other

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./download_and_prepare_commonvoice_shar.sh --lang <lang_code> [options]

Required:
  --lang <lang_code>                  CommonVoice locale (e.g. es, nl, de, zh-CN)

Options:
  --download-dest <path>              Destination for download_commonvoice24.sh
                                      Default: /capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24
  --corpus-dir-hint <path>            Preferred corpus_dir for Lhotse conversion.
                                      Used if <path>/<lang>/train.tsv exists.
                                      Default: /capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24/cv-corpus-24.0-2025-12-05
  --shar-root <path>                  SHAR root directory.
                                      Default: /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR
  --train-name <name>                 Output folder name under stage_2/commonvoice.
                                      Default: <lang>_train
  --other-name <name>                 Output folder name under stage_1/commonvoice.
                                      Default: <lang>_other
  --benchmark-repo <path>             benchmark-audio-tokenizer repo root.
                                      Default: /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer
  --python-bin <path>                 Python executable for conversion.
                                      Default: /opt/venv/bin/python
  --text-tokenizer <path>             tokenizer.json for text tokenization.
                                      Default: /capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json
  --cpu <int>                         CPU cap for conversion (maps to --num_workers).
                                      Default: 288
  --num-workers <int>                 Parallel workers for Lhotse conversion.
                                      Alias of --cpu (kept for compatibility)
  --target-sr <int>                   Target sample rate.
                                      Default: 24000
  --connections <int>                 aria2 connections for download step.
                                      Default: 16
  --skip-download                     Skip download; only run conversion.
  --skip-train                        Skip train split conversion.
  --skip-other                        Skip other split conversion.
  --skip-clean                        Skip clip cleanup/compression step.
  --clean-dry-run                     Show cleanup actions without changing files.
  -h, --help                          Show this help.

Notes:
  - Download runs under conda env "tools".
  - Conversion uses:
      PYTHONPATH=<benchmark-repo> <python-bin> -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar
  - The conversion script must support --shar_output_dir.
EOF
}

LANG=""
DOWNLOAD_DEST="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24"
CORPUS_DIR_HINT="/capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice24/cv-corpus-24.0-2025-12-05"
SHAR_ROOT="/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR"
TRAIN_NAME=""
OTHER_NAME=""
BENCHMARK_REPO="/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-audio-tokenizer"
PYTHON_BIN="/opt/venv/bin/python"
TEXT_TOKENIZER="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok/tokenizer.json"
NUM_WORKERS=288
CPU=""
TARGET_SR=24000
CONNECTIONS=16
SKIP_DOWNLOAD=0
SKIP_TRAIN=0
SKIP_OTHER=0
SKIP_CLEAN=0
CLEAN_DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lang)
      LANG="$2"
      shift 2
      ;;
    --download-dest)
      DOWNLOAD_DEST="$2"
      shift 2
      ;;
    --corpus-dir-hint)
      CORPUS_DIR_HINT="$2"
      shift 2
      ;;
    --shar-root)
      SHAR_ROOT="$2"
      shift 2
      ;;
    --train-name)
      TRAIN_NAME="$2"
      shift 2
      ;;
    --other-name)
      OTHER_NAME="$2"
      shift 2
      ;;
    --benchmark-repo)
      BENCHMARK_REPO="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --text-tokenizer)
      TEXT_TOKENIZER="$2"
      shift 2
      ;;
    --cpu)
      CPU="$2"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --target-sr)
      TARGET_SR="$2"
      shift 2
      ;;
    --connections)
      CONNECTIONS="$2"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-other)
      SKIP_OTHER=1
      shift
      ;;
    --skip-clean)
      SKIP_CLEAN=1
      shift
      ;;
    --clean-dry-run)
      CLEAN_DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$LANG" ]]; then
  echo "ERROR: --lang is required" >&2
  usage
  exit 1
fi

if [[ -n "${CPU}" ]]; then
  NUM_WORKERS="${CPU}"
fi

if [[ -z "${TRAIN_NAME}" ]]; then
  TRAIN_NAME="${LANG}_train"
fi
if [[ -z "${OTHER_NAME}" ]]; then
  OTHER_NAME="${LANG}_other"
fi

if [[ "${TRAIN_NAME}" == */* ]]; then
  echo "ERROR: --train-name must be a folder name, not a path: ${TRAIN_NAME}" >&2
  exit 1
fi
if [[ "${OTHER_NAME}" == */* ]]; then
  echo "ERROR: --other-name must be a folder name, not a path: ${OTHER_NAME}" >&2
  exit 1
fi
if ! [[ "${NUM_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --cpu/--num-workers must be a positive integer: ${NUM_WORKERS}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_commonvoice24.sh"
MANAGE_CLIPS_SCRIPT="${SCRIPT_DIR}/manage_cv_clips.py"
PREPARE_SCRIPT="${BENCHMARK_REPO}/audio_tokenization/utils/prepare_data/prepare_lhotse_recipe_to_shar.py"

if [[ ! -x "$DOWNLOAD_SCRIPT" ]]; then
  echo "ERROR: Download script not found/executable: $DOWNLOAD_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$PREPARE_SCRIPT" ]]; then
  echo "ERROR: Prepare script not found: $PREPARE_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$MANAGE_CLIPS_SCRIPT" ]]; then
  echo "ERROR: Cleanup script not found: $MANAGE_CLIPS_SCRIPT" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TEXT_TOKENIZER" ]]; then
  echo "ERROR: Text tokenizer not found: $TEXT_TOKENIZER" >&2
  exit 1
fi

activate_conda_tools() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH. Cannot activate env 'tools'." >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate tools
}

resolve_corpus_dir() {
  local lang="$1"
  local candidate
  for candidate in \
    "${CORPUS_DIR_HINT}" \
    "${DOWNLOAD_DEST}" \
    "${DOWNLOAD_DEST}/cv-corpus-24.0-2025-12-05"
  do
    if [[ -f "${candidate}/${lang}/train.tsv" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

run_prepare_split() {
  local split="$1"
  local out_dir="$2"
  local corpus_dir="$3"

  mkdir -p "$(dirname "$out_dir")"

  echo ">>> Converting split=${split} language=${LANG}"
  echo "    corpus_dir=${corpus_dir}"
  echo "    output_dir=${out_dir}"

  PYTHONPATH="${BENCHMARK_REPO}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m audio_tokenization.utils.prepare_data.prepare_lhotse_recipe_to_shar \
      --recipe commonvoice \
      --corpus_dir "${corpus_dir}" \
      --split "${split}" \
      --language "${LANG}" \
      --target_sample_rate "${TARGET_SR}" \
      --text_tokenizer "${TEXT_TOKENIZER}" \
      --num_workers "${NUM_WORKERS}" \
      --shar_output_dir "${out_dir}"
}

run_cleanup() {
  local lang_dir="$1"
  local -a clean_args
  clean_args=(
    --lang_dir "${lang_dir}"
    --delete train other
    --compress dev test validated invalidated
  )
  if [[ "${CLEAN_DRY_RUN}" -eq 1 ]]; then
    clean_args+=(--dry_run)
  fi

  echo ">>> Cleaning clips in ${lang_dir}"
  echo "    delete: train other"
  echo "    compress: dev test validated invalidated"
  "${PYTHON_BIN}" "${MANAGE_CLIPS_SCRIPT}" "${clean_args[@]}"
}

echo "=========================================="
echo "CommonVoice language: ${LANG}"
echo "Download destination: ${DOWNLOAD_DEST}"
echo "Corpus dir hint:      ${CORPUS_DIR_HINT}"
echo "SHAR root:            ${SHAR_ROOT}"
echo "Train folder name:    ${TRAIN_NAME}"
echo "Other folder name:    ${OTHER_NAME}"
echo "CPU workers:          ${NUM_WORKERS}"
echo "Cleanup enabled:      $((1-SKIP_CLEAN))"
echo "Cleanup dry run:      ${CLEAN_DRY_RUN}"
echo "=========================================="

if [[ "${SKIP_DOWNLOAD}" -eq 0 ]]; then
  echo ">>> Activating conda env: tools"
  activate_conda_tools
  echo ">>> Downloading CommonVoice locale: ${LANG}"
  CONNECTIONS="${CONNECTIONS}" DEST_DIR="${DOWNLOAD_DEST}" bash "${DOWNLOAD_SCRIPT}" "${LANG}"
else
  echo ">>> Skipping download step"
fi

if ! CORPUS_DIR="$(resolve_corpus_dir "${LANG}")"; then
  echo "ERROR: Could not locate corpus dir containing ${LANG}/train.tsv." >&2
  echo "Checked:" >&2
  echo "  - ${CORPUS_DIR_HINT}" >&2
  echo "  - ${DOWNLOAD_DEST}" >&2
  echo "  - ${DOWNLOAD_DEST}/cv-corpus-24.0-2025-12-05" >&2
  exit 1
fi

TRAIN_OUT="${SHAR_ROOT}/stage_2/commonvoice/${TRAIN_NAME}"
OTHER_OUT="${SHAR_ROOT}/stage_1/commonvoice/${OTHER_NAME}"

if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
  run_prepare_split "train" "${TRAIN_OUT}" "${CORPUS_DIR}"
else
  echo ">>> Skipping train split conversion"
fi

if [[ "${SKIP_OTHER}" -eq 0 ]]; then
  run_prepare_split "other" "${OTHER_OUT}" "${CORPUS_DIR}"
else
  echo ">>> Skipping other split conversion"
fi

if [[ "${SKIP_CLEAN}" -eq 0 ]]; then
  if [[ ! -d "${CORPUS_DIR}/${LANG}" ]]; then
    echo "ERROR: Missing language directory for cleanup: ${CORPUS_DIR}/${LANG}" >&2
    exit 1
  fi
  run_cleanup "${CORPUS_DIR}/${LANG}"
else
  echo ">>> Skipping cleanup step"
fi

echo ""
echo "Done."
echo "Train output: ${TRAIN_OUT}"
echo "Other output: ${OTHER_OUT}"
