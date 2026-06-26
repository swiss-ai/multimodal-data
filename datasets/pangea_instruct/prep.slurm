#!/bin/bash
# Pangea prep: stream JSON→JSONL (filter val2017) + parallel zip→tar (10 zips).
# Single 1-node CPU job, both steps run sequentially inside one srun.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=prep-pangea
#SBATCH --environment=nemo_25_11
#SBATCH --output=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pangea-instruct-prep-%A.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/apertus/multimodal-data/01-dataset-download/logs/pangea-instruct-prep-%A.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal

set -euo pipefail

cd /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/pangea_instruct

echo "==========================================================="
echo "Pangea prep — JSON→JSONL + parallel zip→tar"
echo "==========================================================="

uv pip install --python /opt/venv/bin/python3 ijson

python prep.py --steps jsonl,retar --retar-workers 10

echo
echo "==========================================================="
echo "Done. Artifacts:"
echo "  jsonl: /capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/pangea_instruct/PangeaIns_train.jsonl"
echo "  tars:  /capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___neulab___PangeaInstruct/**/*.tar"
echo "==========================================================="
