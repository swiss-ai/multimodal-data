#!/bin/bash
#SBATCH --job-name=dedup-mint_pdf
#SBATCH --output=logs/dedup_%A_%a.out
#SBATCH --error=logs/dedup_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --array=0-6

export PYTHONUNBUFFERED=1

datasets=(
  "mlfoundations/MINT-1T-PDF-CC-2024-18"
  "mlfoundations/MINT-1T-PDF-CC-2024-10"
  "mlfoundations/MINT-1T-PDF-CC-2023-50"
  "mlfoundations/MINT-1T-PDF-CC-2023-40"
  "mlfoundations/MINT-1T-PDF-CC-2023-23"
  "mlfoundations/MINT-1T-PDF-CC-2023-14"
  "mlfoundations/MINT-1T-PDF-CC-2023-06"
)

array_id=${SLURM_ARRAY_TASK_ID}

echo "Start time: $(date)"
echo "================================"

python3 "$(dirname "$0")/dedup_mint_pdf.py" --dataset "${datasets[$array_id]}" --num_cpus 64

EXIT_CODE=$?

echo "================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
