#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mm-pipeline
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --partition=normal

set -euo pipefail

export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache"
export CACHE_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache"

echo "Start: $(date)"

config_file="$1"
echo "Using config file: $config_file"
echo "Config file contents:"
echo "--- BEGIN ---"
cat "$config_file"
echo "---  END  ---"

source ".venv/bin/activate"
python3 main.py "$config_file"

echo "End: $(date)"
echo "Total elapsed time: $SECONDS seconds"
