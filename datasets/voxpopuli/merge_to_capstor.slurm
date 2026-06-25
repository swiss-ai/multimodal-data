#!/bin/bash
# Merge VoxPopuli 400k (iopsstor) into VoxPopuli 100k (capstor).
#
# Two steps:
#   1. rsync audio files: 23 parallel rsync jobs (one per language)
#   2. Merge VAD results: cat JSONL per language
#
# Capstor structure:  raw_audios/{lang}/*.ogg
# Iopsstor structure: {lang}/*.ogg
#
#SBATCH --account=infra01
#SBATCH --environment=nemo
#SBATCH --job-name=merge-vp400k
#SBATCH --reservation=PA-2338-RL
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/merge-vp400k-%x-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/merge-vp400k-%x-%A.err
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288

set -euo pipefail

SRC_AUDIO="/iopsstor/scratch/cscs/${USER}/audio-datasets/raw/voxpopuli"
DST_AUDIO="/capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/raw_audios"

SRC_VAD="/iopsstor/scratch/cscs/${USER}/audio-datasets/raw/voxpopuli/vad_results"
DST_VAD_CAPSTOR="/capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/vad_results"
DST_VAD_MERGED="/capstor/store/cscs/swissai/infra01/audio-datasets/voxpopuli/vad_results_merged"

LANGS=(bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro sk sl sv)
MAX_PARALLEL=23

mkdir -p "${DST_AUDIO}" "${DST_VAD_MERGED}"

echo "=========================================="
echo "VoxPopuli 400k → capstor merge"
echo "=========================================="
echo "[$(date '+%F %T')] Start"
echo "Source audio: ${SRC_AUDIO}"
echo "Dest audio:   ${DST_AUDIO}"
echo "Languages:    ${#LANGS[@]}"
echo ""

# ---- Step 1: rsync audio (23 parallel jobs) ----
echo "=== Step 1: rsync audio files ==="

rsync_lang() {
    local lang="$1"
    local src="${SRC_AUDIO}/${lang}/"
    local dst="${DST_AUDIO}/${lang}/"

    if [ ! -d "${src}" ]; then
        echo "[${lang}] Source not found: ${src}, skipping"
        return 0
    fi

    mkdir -p "${dst}"
    echo "[$(date '+%F %T')] [${lang}] rsync start"
    rsync -a --omit-dir-times --info=progress2 "${src}" "${dst}"
    echo "[$(date '+%F %T')] [${lang}] rsync done"
}
export -f rsync_lang
export SRC_AUDIO DST_AUDIO

printf '%s\n' "${LANGS[@]}" | xargs -P "${MAX_PARALLEL}" -I{} bash -c 'rsync_lang "$@"' _ {}

echo ""
echo "[$(date '+%F %T')] All rsync jobs done"

# ---- Step 2: merge VAD results per language ----
echo ""
echo "=== Step 2: merge VAD results ==="

for lang in "${LANGS[@]}"; do
    capstor_vad="${DST_VAD_CAPSTOR}/voxpopuli_${lang}.jsonl"
    iopsstor_vad="${SRC_VAD}/voxpopuli_${lang}.jsonl"
    merged="${DST_VAD_MERGED}/voxpopuli_${lang}.jsonl"

    if [ -f "${capstor_vad}" ] && [ -f "${iopsstor_vad}" ]; then
        # Both exist: concatenate (capstor 100k entries first, then iopsstor extras)
        cat "${capstor_vad}" "${iopsstor_vad}" > "${merged}"
        capstor_n=$(wc -l < "${capstor_vad}")
        iopsstor_n=$(wc -l < "${iopsstor_vad}")
        merged_n=$(wc -l < "${merged}")
        echo "[${lang}] Merged: ${capstor_n} (capstor) + ${iopsstor_n} (iopsstor) = ${merged_n} entries"
    elif [ -f "${iopsstor_vad}" ]; then
        # Only iopsstor (new language: bg, da, el, lt, lv, mt, pt, sv)
        cp "${iopsstor_vad}" "${merged}"
        n=$(wc -l < "${merged}")
        echo "[${lang}] Copied from iopsstor: ${n} entries (new language)"
    elif [ -f "${capstor_vad}" ]; then
        # Only capstor (shouldn't happen for 400k, but handle gracefully)
        cp "${capstor_vad}" "${merged}"
        n=$(wc -l < "${merged}")
        echo "[${lang}] Copied from capstor: ${n} entries (iopsstor VAD missing)"
    else
        echo "[${lang}] WARNING: no VAD results found in either location"
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Audio (capstor):"
for lang in "${LANGS[@]}"; do
    dir="${DST_AUDIO}/${lang}"
    if [ -d "${dir}" ]; then
        count=$(find "${dir}" -name '*.ogg' | wc -l)
        printf "  %-5s %'10d files\n" "${lang}" "${count}"
    fi
done

echo ""
echo "VAD merged:"
for f in "${DST_VAD_MERGED}"/*.jsonl; do
    count=$(wc -l < "$f")
    printf "  %-35s %'10d entries\n" "$(basename "$f")" "${count}"
done

echo ""
echo "[$(date '+%F %T')] All done"
