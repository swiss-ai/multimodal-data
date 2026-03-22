#!/bin/bash
# Download + convert CommonVoice language(s) to SHAR format (job array)
#
# Usage:
#   LANGS="eo be" sbatch prepare_commonvoice_shar.slurm
#   LANGS="eo be gl eu id fa" sbatch prepare_commonvoice_shar.slurm
#
# Each array task runs the full pipeline for one language:
#   download -> SHAR train -> SHAR other -> cleanup
#
#SBATCH --account=infra01
#SBATCH --job-name=cv-shar
#SBATCH --reservation=PA-2338-RL
#SBATCH --environment=nemo
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/commonvoice24/logs/%x-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/commonvoice24/logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --exclusive

set -euo pipefail

SCRIPT_DIR="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/commonvoice24"
PIPELINE="${SCRIPT_DIR}/download_and_prepare_commonvoice_shar.sh"

# Split LANGS env var into array
IFS=' ' read -ra LANG_ARRAY <<< "${LANGS:?ERROR: set LANGS env var, e.g. LANGS=\"eo be\" sbatch $0}"

LANG_CODE="${LANG_ARRAY[$SLURM_ARRAY_TASK_ID]}"

echo "=========================================="
echo "CommonVoice SHAR Pipeline (array job)"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Node:         $(hostname)"
echo "Language:     ${LANG_CODE}"
echo "CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "=========================================="

bash "${PIPELINE}" --lang "${LANG_CODE}" --cpu "${SLURM_CPUS_PER_TASK}" --skip-download

echo "=========================================="
echo "Done: ${LANG_CODE} ($(date))"
echo "=========================================="
