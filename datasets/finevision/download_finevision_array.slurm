#!/bin/bash
# Slurm array job — one task per FineVision subset.
# 12 nodes downloading in parallel, each with its own network IP → no shared
# bandwidth contention. Uses the isolated hftools venv with hf-xet for
# Xet-backend chunked-parallel transfers.
#
# Each task:
#   - Skips if $DEST/$s/.complete exists (idempotent across re-runs).
#   - Calls `hf download` which itself is idempotent at the file level
#     (will resume any partial subset state from previous downloads).
#   - Touches $DEST/$s/.complete on success.
#
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=finevision-dl
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision/logs/array_%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision/logs/array_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --partition=normal
#SBATCH --array=0-40%12

set -euo pipefail

SUBSETS=(
    # ── Tier 1: clearly permissive ─────────────────────────────────────────
    "DoclingMatix"               # 0  — MIT, 951 GB
    # Synth{ChartNet,CodeNet,FormulaNet} — ALL DROPPED: IBM Granite Docling
    # format (<chart|code|formula><loc_0-500>) incompatible with our BLIP-style
    # grounding convention. Data physically removed from disk.
    "cocoqa"                     # 4  — CC-BY-4.0
    "densefusion_1m"             # 5  — CC-BY-4.0, 145 GB
    "localized_narratives"       # 6  — CC-BY-4.0
    "nlvr2"                      # 7  — CC-BY-4.0
    # funsd — DROPPED: 5.2% filter retention × 37% duplicate-turn = ~no useful rows
    # synthdog — DROPPED: 9.4% filter retention; OCR judge can't validate transcription
    "multihiertt"                # 10 — MIT
    "art"                        # 11 — CC-BY-4.0
    "wordart"                    # 12 — CC-BY-4.0
    "lnqa"                       # 13 — CC-BY-4.0, 266 GB
    "CoSyn_400k_chemical"        # 14 — CC-BY-4.0
    "CoSyn_400k_circuit"         # 15 — CC-BY-4.0
    "CoSyn_400k_document"        # 16 — CC-BY-4.0
    "CoSyn_400k_graphic"         # 17 — CC-BY-4.0
    "CoSyn_400k_math"            # 18 — CC-BY-4.0
    "CoSyn_400k_music"           # 19 — CC-BY-4.0
    "CoSyn_400k_nutrition"       # 20 — CC-BY-4.0
    "CoSyn_400k_table"           # 21 — CC-BY-4.0
    # ── GPT-distilled but explicitly permissive on HF ───────────────────────
    "mmevol"                     # 22 — Apache-2.0
    # groundui — DROPPED: redundant with molmopoint_guisyn (UI pointing/grounding)
    "aguvis-stage-1"             # 24 — Apache-2.0, 228 GB
    "chinesememe"                # 25 — Apache-2.0
    "memotion"                   # 26 — Apache-2.0
    # ── Audit additions ─────────────────────────────────────────────────────
    "olmOCR-mix-0225-books"      # 27 — ODC-BY
    "olmOCR-mix-0225-documents"  # 28 — ODC-BY, 97 GB
    "tat_dqa"                    # 29 — CC-BY-4.0
    "svrd"                       # 30 — CC-BY-4.0
    "yesbut"                     # 31 — Apache-2.0
    "mmra"                       # 32 — MIT
    "latex_handwritten"          # 33 — Apache-2.0
    "tal_ocr_eng"                # 34 — Apache-2.0
    "coco_colors"                # 35 — CC-BY-4.0 (COCO-derived)
    "spot_the_diff"              # 36 — CC-BY-4.0
    "latexformulas"              # 37 — OpenRAIL (training allowed)
    "spark"                      # 38 — OpenRAIL
    "handwriting_forms"          # 39 — OpenRAIL
    "wildvision"                 # 40 — CC-BY-4.0 (WildVision/wildvision-arena-data)
    # ── DROPPED after license audit (kept here as a tombstone) ──────────────
    # slidevqa      — NTT Software Evaluation License (research-only)
    # pdfvqa        — no license, scraped PDF images
    # spatialsense  — BSD-2 code + NYU/Flickr images (mixed restrictions)
    # k12_printing  — no license tag
    # maptext       — ICDAR competition data, commercial use unclear
    # captcha       — no license tag, source unclear
    # vqaonbd       — "© All Rights Reserved", FinTabNet derivative (CC-BY-SA)
    # indoor_qa     — MIT Indoor 67 source (research-only)
)

s="${SUBSETS[$SLURM_ARRAY_TASK_ID]}"
DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision
HF=/capstor/scratch/cscs/xyixuan/venvs/hftools/bin/hf

mkdir -p "$DEST" /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision/logs

echo "=========================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] subset=$s node=$(hostname -s)"
echo "=========================================="

if [ -f "$DEST/$s/.complete" ]; then
    echo "already complete, exiting"
    exit 0
fi

# Xet replaces hf_transfer in newer huggingface_hub; HF_XET_HIGH_PERFORMANCE
# raises concurrency + buffer sizes for high-RAM, high-bandwidth nodes.
export HF_XET_HIGH_PERFORMANCE=1

START=$(date +%s)
"$HF" download HuggingFaceM4/FineVision \
    --repo-type=dataset \
    --include "${s}/*" \
    --local-dir "$DEST"

touch "$DEST/$s/.complete"
END=$(date +%s)
SIZE=$(du -sh "$DEST/$s" | awk '{print $1}')
echo "=========================================="
echo "$s: $SIZE downloaded in $((END - START))s"
echo "=========================================="
