#!/usr/bin/env bash
# MLS -> SHAR submission examples (copy/paste).
# This script prints example sbatch commands; it does not submit jobs.

set -euo pipefail

SLURM_SCRIPT="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/mls/prepare_multilingual_to_shar.slurm"

cat <<EOF
# ==========================================================
# MLS -> SHAR examples
# ==========================================================
# Script:
#   ${SLURM_SCRIPT}
#
# Notes:
# - English is array index 0 in default LANGUAGES order.
# - Default LANGUAGES:
#   "english dutch french german italian polish portuguese spanish"
# - Output pattern:
#   /capstor/store/cscs/swissai/infra01/audio-datasets/SHAR/stage_2/mls_<language>_<split>

# 1) English train only (recommended start)
sbatch --array=0 \\
  --export=ALL,LANGUAGES="english",SPLITS="train",NUM_WORKERS=128 \\
  "${SLURM_SCRIPT}"

# 2) English train/dev/test
sbatch --array=0 \\
  --export=ALL,LANGUAGES="english",SPLITS="train dev test",NUM_WORKERS=128 \\
  "${SLURM_SCRIPT}"

# 3) All 8 languages train only (english + 7 multilingual MLS)
sbatch \\
  --export=ALL,SPLITS="train",NUM_WORKERS=128 \\
  "${SLURM_SCRIPT}"

# 4) Non-English only, train split
sbatch --array=0-6 \\
  --export=ALL,LANGUAGES="dutch french german italian polish portuguese spanish",SPLITS="train",NUM_WORKERS=64 \\
  "${SLURM_SCRIPT}"

# 5) Non-English all available splits (train/dev/test/1_hours/9_hours)
sbatch --array=0-6 \\
  --export=ALL,LANGUAGES="dutch french german italian polish portuguese spanish",SPLITS="train dev test 1_hours 9_hours",NUM_WORKERS=64 \\
  "${SLURM_SCRIPT}"

# 6) One selected multilingual language (example: german)
sbatch --array=0 \\
  --export=ALL,LANGUAGES="german",SPLITS="train",NUM_WORKERS=64 \\
  "${SLURM_SCRIPT}"
EOF

