#!/bin/bash
# Launch Qwen3.5-397B-A17B BF16 as TWO replicas behind a load-balancing router,
# via sml + sglang on 8 GH200 nodes (2 workers × 4 nodes each, TP=16 per
# worker). Single OpenAI-compatible endpoint via the router.
#
# Usage (login or compute node with sbatch access):
#   source /iopsstor/scratch/cscs/xyixuan/apertus/model-serving/.venv/bin/activate
#   bash launch_qwen35_captioner_2x.sh
#
# 2x the throughput of the single-instance launcher in the same directory.

set -euo pipefail

export SML_FIRECREST_SYSTEM=clariden
export SML_PARTITION=normal
export SML_RESERVATION=SD-69241-apertus-1-5

cd /iopsstor/scratch/cscs/xyixuan/apertus/model-launch

sml advanced \
  --slurm-nodes 8 \
  --slurm-workers 2 \
  --slurm-nodes-per-worker 4 \
  --use-router \
  --disable-ocf \
  --serving-framework sglang \
  --slurm-time 12:00:00 \
  --slurm-environment "$(pwd)/src/swiss_ai_model_launch/assets/envs/sglang.toml" \
  --pre-launch-cmds "pip install --no-cache-dir nvidia-cudnn-cu12==9.16.0.29" \
  --framework-args "--model-path /capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.5-397B-A17B \
    --host 0.0.0.0 --port 5000 --tp-size 16 \
    --trust-remote-code \
    --dist-timeout 3600 \
    --served-model-name Qwen/Qwen3.5-397B-A17B-$(whoami)"
