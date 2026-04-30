#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5
#SBATCH --job-name=qwen-ocr-serve-10n
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/model-launch/src/swiss_ai_model_launch/assets/envs/vllm.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen_ocr_serve_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen_ocr_serve_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=36
#SBATCH --partition=normal
#SBATCH --exclusive
#
# Spawn 20 vLLM instances of Qwen3.6-27B (TP=2 each) across 10 GH200 nodes,
# 2 instances per node (GPUs 0,1 → port 8080; GPUs 2,3 → port 8081).
# No router — client distributes work round-robin to the 20 endpoints.

set -e

MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.6-27B
ENDPOINTS_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen_ocr_endpoints_${SLURM_JOB_ID}
mkdir -p "${ENDPOINTS_DIR}"
echo "endpoints dir: ${ENDPOINTS_DIR}"
echo "nodelist: ${SLURM_JOB_NODELIST}"

srun -u --kill-on-bad-exit=0 bash -c '
set -e
PORT=$((8080 + SLURM_LOCALID))
HOST=$(hostname -s)
echo "[$(date +%T)] task ${SLURM_PROCID} on ${HOST} localid=${SLURM_LOCALID} port=${PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Announce endpoint (atomic create per task)
echo "${HOST}:${PORT}" > "'"${ENDPOINTS_DIR}"'/${SLURM_PROCID}.endpoint"

vllm serve '"${MODEL}"' \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served-model-name Qwen/Qwen3.6-27B-xyixuan \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --trust-remote-code
'
