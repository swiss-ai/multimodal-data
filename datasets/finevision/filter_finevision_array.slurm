#!/bin/bash
# Filter FineVision subsets by per-turn quality ratings.
# One task per subset; runs the Python filter that drops rows with
# visual_dependency_min < 2 OR image_correspondence_min < 2 (factual-quality
# floor — the two dimensions where 1/5 means "actively wrong" or "wastes
# image tokens"). relevance_min / formatting_min are left unfiltered per
# the FineVision paper's recommendation.
#
# Output: /capstor/.../processed/sft/finevision/{subset}/train-*.parquet
# Schema: images, texts, source (rating columns dropped — only needed for
# the filter step itself).
#
# Idempotent: skips per-shard if the output parquet already exists.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=finevision-filter
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision/logs/filter_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision/logs/filter_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --array=0-40%12

set -euo pipefail

SUBSETS=(
    "DoclingMatix"               # 0
    # Synth{ChartNet,CodeNet,FormulaNet} — DROPPED (incompatible Docling format)
    "cocoqa"                     # 4
    "densefusion_1m"             # 5
    "localized_narratives"       # 6
    "nlvr2"                      # 7
    # funsd — DROPPED (5.2% retention × 37% dup)
    # synthdog — DROPPED (9% retention, OCR judge unreliable)
    "multihiertt"                # 10
    "art"                        # 11
    "wordart"                    # 12
    "lnqa"                       # 13
    "CoSyn_400k_chemical"        # 14
    "CoSyn_400k_circuit"         # 15
    "CoSyn_400k_document"        # 16
    "CoSyn_400k_graphic"         # 17
    "CoSyn_400k_math"            # 18
    "CoSyn_400k_music"           # 19
    "CoSyn_400k_nutrition"       # 20
    "CoSyn_400k_table"           # 21
    "mmevol"                     # 22
    # groundui — DROPPED (redundant with molmopoint_guisyn)
    "aguvis-stage-1"             # 24
    "chinesememe"                # 25
    "memotion"                   # 26
    "olmOCR-mix-0225-books"      # 27
    "olmOCR-mix-0225-documents"  # 28
    "tat_dqa"                    # 29
    "svrd"                       # 30
    "yesbut"                     # 31
    "mmra"                       # 32
    "latex_handwritten"          # 33
    "tal_ocr_eng"                # 34
    "coco_colors"                # 35
    "spot_the_diff"              # 36
    "latexformulas"              # 37
    "spark"                      # 38
    "handwriting_forms"          # 39
    "wildvision"                 # 40
)

s="${SUBSETS[$SLURM_ARRAY_TASK_ID]}"
SRC=/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision/$s

echo "=========================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] subset=$s node=$(hostname -s)"
echo "=========================================="

if [ ! -f "$SRC/.complete" ]; then
    echo "skip: download not complete for $s"
    exit 0
fi

mkdir -p /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision/logs

cd /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision

START=$(date +%s)
python filter_finevision.py "$s"
END=$(date +%s)
echo "=========================================="
echo "$s filtered in $((END - START))s"
echo "=========================================="
