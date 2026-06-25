#!/bin/bash
#SBATCH --job-name=rsync-audio-only
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/rsync_audio_only_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/logs/rsync_audio_only_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=xfer
#SBATCH --account=infra01
#SBATCH --ntasks=1

SRC="/iopsstor/scratch/cscs/xyixuan/audio-datasets/audio_only_shar"
DST="/capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_1"

DATASETS=(
    audioset_bal_train_audio_only
    audioset_unbal_train_audio_only
    gemeinderat_audio_only
    mtg_jamendo_train
    peoples_speech_clean_shar_audio_only
    peoples_speech_dirty_shar_audio_only
    suno_s1_shar
    ups_all_lang_shar_v2
    voxpopuli_shar
)

for ds in "${DATASETS[@]}"; do
    echo "===== rsync $ds ====="
    echo "Start: $(date)"
    rsync -a --whole-file "${SRC}/${ds}/" "${DST}/${ds}/"
    echo "Done:  $(date)"
    echo ""
done
