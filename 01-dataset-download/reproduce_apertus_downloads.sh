#!/bin/bash
# Commands used to download the multimodal datasets for Apertus 1.5 training.
# All downloads use `huggingface-cli download` (raw files -> --local-dir, pinned
# by --revision) with HF_HUB_ENABLE_HF_TRANSFER=1.
set -euo pipefail

export PYTHONPATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages:${PYTHONPATH:-}"
export PATH="/capstor/store/cscs/swissai/infra01/MLLM/pip-packages/bin:${PATH}"
export HF_HUB_ENABLE_HF_TRANSFER=1

# 1) Paired Img/Txt data — https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M
LLAVA_DEST="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/hf___mvp-lab___LLaVA-OneVision-1.5-Mid-Training-85M"
HF_HUB_CACHE="${LLAVA_DEST}/.hf_cache" \
huggingface-cli download "mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M" \
    --repo-type dataset \
    --revision "c5218cad785eba7d218137e8ce4997bda568a050" \
    --local-dir "${LLAVA_DEST}" \
    --max-workers 64
rm -rf "${LLAVA_DEST}/.hf_cache"
