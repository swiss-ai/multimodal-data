#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mm-pipeline
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=24:00:00
#SBATCH --partition=normal

set -euo pipefail

export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache"
export CACHE_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache"

echo "Start: $(date)"

source ".venv/bin/activate"

config_files=(
  "configs/medtrinity_demo.json"
  "configs/medmax.json"
  "configs/mmc4.json"
)

for config_file in "${config_files[@]}"; do
  echo ""
  echo "========================================="
  echo "Using config file: $config_file"
  echo "Config file contents:"
  cat "$config_file"
  echo "========================================="

  python3 main.py "$config_file"

  echo "Completed run with config file: $config_file"
done

echo "End: $(date)"
echo "Total elapsed time: $SECONDS seconds"
