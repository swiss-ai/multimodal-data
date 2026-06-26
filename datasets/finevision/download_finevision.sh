#!/bin/bash
# Download 34 FineVision subsets selected for Apertus SFT:
#   - 24 original Tier 1 (permissive, non-overlapping with llava-ov + nemotron-v3)
#   -  9 GPT-distilled-but-permissive (we don't distill from those directly)
#   -  1 English meme (memotion)
#
# Target: /capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision/{subset}/
# Tool: hf CLI from isolated venv (with hf_transfer for fast parallel chunks).

set -e

export HF_HUB_ENABLE_HF_TRANSFER=1
HF=/capstor/scratch/cscs/xyixuan/venvs/hftools/bin/hf
DEST=/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision

SUBSETS=(
    # Tier 1: clearly permissive (CC-BY/Apache/MIT)
    "DoclingMatix"
    "SynthChartNet"
    "SynthCodeNet"
    "SynthFormulaNet"
    "cocoqa"
    "densefusion_1m"
    "localized_narratives"
    "nlvr2"
    "funsd"
    "synthdog"
    "multihiertt"
    "art"
    "wordart"
    "lnqa"
    "CoSyn_400k_chemical"
    "CoSyn_400k_circuit"
    "CoSyn_400k_document"
    "CoSyn_400k_graphic"
    "CoSyn_400k_math"
    "CoSyn_400k_music"
    "CoSyn_400k_nutrition"
    "CoSyn_400k_table"
    # GPT-distilled but permissive (user OK'd: we don't distill from those directly)
    "allava_laion"
    "allava_vflan"
    "mmc_instruct"
    "mmevol"
    "vision_flan(filtered)"
    "mavis_math_rule_geo"
    "groundui"
    "aguvis-stage-1"
    "chinesememe"
    # English memes
    "memotion"
    # ── Audit additions (first-principles pass) ──
    # Clearly permissive (CC-BY / Apache / MIT / ODC-BY):
    "olmOCR-mix-0225-books"
    "olmOCR-mix-0225-documents"
    "tat_dqa"
    "svrd"
    "yesbut"
    "mmra"
    "latex_handwritten"
    "tal_ocr_eng"
    "coco_colors"
    "spot_the_diff"
    # OpenRAIL — training is allowed; use restrictions apply to model deployment
    "latexformulas"
    "spark"
    "handwriting_forms"
    # Likely permissive — verify upstream before relying for redistribution
    "slidevqa"
    "pdfvqa"
    "spatialsense"
    "k12_printing"
    "maptext"
    "captcha"
    "vqaonbd"
    "indoor_qa"
    "wildvision"
)

mkdir -p "$DEST"
N=${#SUBSETS[@]}
i=0
for s in "${SUBSETS[@]}"; do
    i=$((i+1))
    if [ -f "$DEST/$s/.complete" ]; then
        echo "[$i/$N] $s — already complete, skipping"
        continue
    fi
    echo "[$i/$N] $s — downloading (resumes any partial state)"
    "$HF" download HuggingFaceM4/FineVision \
        --repo-type=dataset \
        --include "${s}/*" \
        --local-dir "$DEST" \
        2>&1 | tail -3
    touch "$DEST/$s/.complete"
done
echo
echo "=== summary ==="
for s in "${SUBSETS[@]}"; do
    if [ -d "$DEST/$s" ]; then
        sz=$(du -sh "$DEST/$s" 2>/dev/null | awk '{print $1}')
        nf=$(find "$DEST/$s" -type f | wc -l)
        printf "  %-32s %8s  %4d files\n" "$s" "$sz" "$nf"
    else
        printf "  %-32s MISSING\n" "$s"
    fi
done
