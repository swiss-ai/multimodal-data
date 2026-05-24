#!/bin/bash
#
#SBATCH --job-name=ingest
#SBATCH --account=infra01
#SBATCH --partition=xfer
#SBATCH --time=12:00:00
#SBATCH --output=/path/to/data/scratch/cscs/%u/logs/%j.out
#SBATCH --error=/path/to/data/scratch/cscs/%u/logs/%j.err

if [ -z "${TARGET_DIR}" ]; then
  echo "ERROR: TARGET_DIR environment variable is not set"
  exit 1
fi

#
# Configuration
#

dest="${TARGET_DIR}/pmc_oa"

export RCLONE_CONFIG_PMC_TYPE="http"
export RCLONE_CONFIG_PMC_URL="https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/"

#
# Execute ingestion
#

echo "Starting sync at $(date)."

rclone sync pmc: "$dest" \
  --transfers=8 \
  --include "/oa_comm/txt/**" \
  --include "/oa_other/txt/**" \
  --include "/oa_noncomm/txt/**"

echo "Sync completed at $(date)."
