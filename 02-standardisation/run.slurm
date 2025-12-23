#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=mm-pipeline
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=normal

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch run.slurm <config-file>"
    exit 1
fi

config_file="$1"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

echo "Using config file: $config_file"
echo "Config file contents:"
echo "--- BEGIN ---"
cat "$config_file"
echo "---  END  ---"

uv run main.py "$config_file"

echo "End: $(date)"
