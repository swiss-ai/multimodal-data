#!/bin/bash
# Submit per-language UPS SHAR conversion jobs.
# Each language gets ceil(num_vad_files / TARS_PER_NODE) array tasks.
#
# Usage:
#   bash submit_per_lang.sh                # submit all 88 languages
#   bash submit_per_lang.sh en es ar de    # submit specific languages
#   bash submit_per_lang.sh --dry-run      # show what would be submitted

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/prepare_vad_to_shar_per_lang.slurm"
VAD_ROOT=/capstor/store/cscs/swissai/infra01/audio-datasets/raw/unsupervised_peoples_speech_commercial_wds/vad_per_lang

# ~262 total nodes across all languages
TARS_PER_NODE=300

DRY_RUN=false
LANGS=()

for arg in "$@"; do
  if [ "$arg" = "--dry-run" ]; then
    DRY_RUN=true
  else
    LANGS+=("$arg")
  fi
done

# Default: all languages
if [ "${#LANGS[@]}" -eq 0 ]; then
  mapfile -t LANGS < <(ls "${VAD_ROOT}" | sort)
fi

echo "Languages to submit: ${#LANGS[@]}"
echo "Tars per node: ${TARS_PER_NODE}"
echo ""

for lang in "${LANGS[@]}"; do
  vad_dir="${VAD_ROOT}/${lang}"
  if [ ! -d "${vad_dir}" ]; then
    echo "SKIP ${lang}: no VAD dir"
    continue
  fi

  num_files=$(ls "${vad_dir}"/*.jsonl 2>/dev/null | wc -l)
  if [ "${num_files}" -eq 0 ]; then
    echo "SKIP ${lang}: no VAD files"
    continue
  fi

  num_nodes=$(( (num_files + TARS_PER_NODE - 1) / TARS_PER_NODE ))
  array_max=$(( num_nodes - 1 ))

  if [ "${DRY_RUN}" = true ]; then
    printf "%-4s  %5d files  %2d nodes  --array=0-%d\n" "${lang}" "${num_files}" "${num_nodes}" "${array_max}"
  else
    job_id=$(sbatch --export=ALL,TARGET_LANG="${lang}" --array="0-${array_max}" --job-name="ups-${lang}" "${SLURM_SCRIPT}" | awk '{print $NF}')
    printf "%-4s  %5d files  %2d nodes  job=%s\n" "${lang}" "${num_files}" "${num_nodes}" "${job_id}"
  fi
done
