#!/bin/bash
# Download all CommonVoice 25.0 locales and emit per-split parquet, one
# locale per node. Each array task: Mozilla DC API → aria2c download →
# pigz-decompressed tar stream → 6 per-split parquet files (no clips/
# directory ever materialized).
#
# Storage strategy: tar download + parquet emission write directly to
# capstor (durable, shared). Empirically capstor handles bulk sequential
# writes well (~1.8 GB/s on a 89 GB cp test) and the iopsstor staging
# round-trip wasn't paying off for these dataset sizes.
#
# Submission patterns:
#
#   # Run all 53 locales in parallel (one node each):
#   sbatch --array=0-52 download_cv_parquet.slurm
#
#   # Cap concurrency to e.g. 8 nodes at a time (polite to Mozilla DC):
#   sbatch --array=0-52%8 download_cv_parquet.slurm
#
#   # Run a subset:
#   LANGS="zh-CN ar de" sbatch --array=0-2 download_cv_parquet.slurm
#
#   # Reuse an existing tarball on disk (no Mozilla traffic):
#   LANGS="zh-TW" SKIP_DOWNLOAD=1 \
#       TAR_DIR=/iopsstor/scratch/cscs/${USER}/cv25_tmp/tar \
#       sbatch --array=0 download_cv_parquet.slurm
#
#   # Keep tarballs after parquet emission (re-runnable without re-download):
#   KEEP_TAR=1 sbatch --array=0-52 download_cv_parquet.slurm
#
#SBATCH --account=infra01
#SBATCH --job-name=cv25-parquet
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --environment=nemo_26_02
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/commonvoice25/logs/%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/commonvoice25/logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

# Default locale list = the 52 v24-baseline langs + gsw (Swiss German /
# Alsatian, new in v25). DATASETS dict in the Python script must contain
# every locale listed here. Override via LANGS env var for subsets or to
# add new v25 locales.
DEFAULT_LANGS=(
    ar sq hy-AM eu gsw
    be bg ca cs da
    nl en eo et fi
    fr gl ka de el
    hu is ga-IE it ja
    ko lv lt mk mt
    nb-NO nn-NO pl pt ro
    ru sr sk sl es
    sv-SE tr uk cy zh-CN
    zh-HK zh-TW rm-sursilv th vi
    yue id fa kk
    hi bn ur ta te mr
)

if [[ -n "${LANGS:-}" ]]; then
    IFS=' ' read -ra LANG_ARRAY <<< "${LANGS}"
else
    LANG_ARRAY=("${DEFAULT_LANGS[@]}")
fi

if [[ "${SLURM_ARRAY_TASK_ID:-NONE}" == "NONE" ]]; then
    echo "ERROR: this script must be submitted as an array job (sbatch --array=...)" >&2
    exit 1
fi
if (( SLURM_ARRAY_TASK_ID >= ${#LANG_ARRAY[@]} )); then
    echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} out of range for ${#LANG_ARRAY[@]} langs" >&2
    exit 1
fi

LANG_CODE="${LANG_ARRAY[$SLURM_ARRAY_TASK_ID]}"

# ---------------------------------------------------------------------------
# Storage layout — tar + parquet both land directly on capstor.
# ---------------------------------------------------------------------------
# TAR_DIR (capstor):  raw tarball cache. Auto-deleted after successful
#   parquet emission unless KEEP_TAR=1.
# OUT_ROOT (capstor): per-locale per-split parquet output (durable).

OUT_ROOT="${OUT_ROOT:-/capstor/store/cscs/swissai/infra01/audio-datasets/processed/commonvoice25}"
TAR_DIR="${TAR_DIR:-/capstor/store/cscs/swissai/infra01/audio-datasets/raw/commonvoice25}"
TOKEN_FILE="${MOZILLA_DC_TOKEN_FILE:-/iopsstor/scratch/cscs/${USER}/.mozilla_dc_token}"

SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/audio/commonvoice25"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
SCRIPT="${SCRIPT_DIR}/download_cv_to_parquet.py"

# pigz lives in the user's bin on this cluster. Without it the script falls
# back to Python's zlib (correct, just ~30% slower). Make it discoverable
# under the container env.
export PATH="/users/${USER}/bin:${PATH}"
export MOZILLA_DC_TOKEN_FILE="${TOKEN_FILE}"

OUT_DIR="${OUT_ROOT}/${LANG_CODE}"
TAR_PATH="${TAR_DIR}/cv25_${LANG_CODE}.tar.gz"

mkdir -p "${OUT_DIR}" "${TAR_DIR}"

echo "=========================================="
echo "CommonVoice 25 → parquet (array job)"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Array task:    ${SLURM_ARRAY_TASK_ID}"
echo "Node:          $(hostname)"
echo "Locale:        ${LANG_CODE}"
echo "Out dir:       ${OUT_DIR}        (capstor)"
echo "Tar cache:     ${TAR_PATH}"
echo "CPUs:          ${SLURM_CPUS_PER_TASK}"
echo "Reservation:   ${SLURM_JOB_RESERVATION:-none}"
echo "Skip dl:       ${SKIP_DOWNLOAD:-0}"
echo "Keep tar:      ${KEEP_TAR:-0}"
echo "Shard rows:    ${SHARD_ROWS:-0}"
echo "Started:       $(date)"
echo "=========================================="

declare -a EXTRA_ARGS=()
if [[ "${SKIP_DOWNLOAD:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--skip-download --tar-path "${TAR_PATH}")
else
    EXTRA_ARGS+=(--tar-path "${TAR_PATH}")
fi

if [[ "${KEEP_TAR:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--keep-tar)
fi

if [[ -n "${SHARD_ROWS:-}" ]] && [[ "${SHARD_ROWS}" -gt 0 ]]; then
    EXTRA_ARGS+=(--shard-rows "${SHARD_ROWS}")
fi

# ---------------------------------------------------------------------------
# Run: download tar + parquet emission, both directly to capstor.
# ---------------------------------------------------------------------------
echo "[$(date)] === download + parquet emission ==="
"${PYTHON_BIN}" "${SCRIPT}" \
    --lang "${LANG_CODE}" \
    --output-dir "${OUT_DIR}" \
    "${EXTRA_ARGS[@]}"

echo "=========================================="
echo "Done: ${LANG_CODE} ($(date))"
echo "Final parquet output: ${OUT_DIR}"
echo "=========================================="
