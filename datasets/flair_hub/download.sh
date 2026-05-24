#!/bin/bash
#SBATCH --job-name=dwnldIGNtiles
#SBATCH --output=logs/ign_tiles-%A.out
#SBATCH --error=logs/ign_tiles-%A.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=normal

# ========================================
# IGN PLANIGNV2 Satellite Tile Downloader
# ========================================
# Downloads IGN aerial/satellite tiles for French cities using img2dataset.
#
# Step 1: Generate URL CSV (run separately):
#   python main.py
#
# Step 2: Submit this script to download tiles:
#   sbatch download.sh
#
# Configuration (environment variables):
#   INPUT_CSV    - CSV with tile URLs (default: france_tiles_template.csv)
#   OUTPUT_DIR   - Output directory for webdataset shards
#   PROCESS_COUNT - Number of img2dataset processes (default: 1)
# ========================================

VENV_PATH="${VENV_PATH:-$HOME/downloads/.venv}"
source "$VENV_PATH/bin/activate"

export NO_ALBUMENTATIONS_UPDATE=1

INPUT_CSV="${INPUT_CSV:-france_tiles_template.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/path/to/data/vision-datasets/ign_city_tiles}"

echo "========================================"
echo "IGN Tile Download Job"
echo "========================================"
echo "Start time: $(date)"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Input CSV: $INPUT_CSV"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

python download.py

EXIT_CODE=$?

echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
