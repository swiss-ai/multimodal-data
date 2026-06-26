#!/bin/bash
#SBATCH --account=infra01
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --job-name=qwen36-serve-40n
#SBATCH --environment=/iopsstor/scratch/cscs/xyixuan/apertus/model-launch/src/swiss_ai_model_launch/assets/envs/vllm.toml
#SBATCH --output=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen36_serve_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen36_serve_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=36
#SBATCH --partition=normal
#SBATCH --exclusive
#
# 80 vLLM Qwen3.6-27B endpoints (TP=2 each) across 40 GH200 nodes,
# 2 instances per node (GPUs 0-1 → port 8080; GPUs 2-3 → port 8081).
# No router, no DP — client distributes work round-robin over the 80
# endpoints published in $ENDPOINTS_DIR.
#
# Proven pattern (mirrors smithsonian/ocr_pass/serve_qwen.slurm at 2x scale).
# DEFAULT CUDA graph capture (no --enforce-eager) — works because each
# instance is independent (TP=2 only, no inter-instance NCCL during capture).

set -e

MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.6-27B
ENDPOINTS_DIR=/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen36_endpoints_${SLURM_JOB_ID}
mkdir -p "${ENDPOINTS_DIR}"
echo "endpoints dir: ${ENDPOINTS_DIR}"
echo "nodelist: ${SLURM_JOB_NODELIST}"

# Convenience: a stable symlink so clients can always find the latest endpoints dir
ln -snf "${ENDPOINTS_DIR}" /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen36_endpoints_latest

srun -u --kill-on-bad-exit=0 bash -c '
set -e
PORT=$((8080 + SLURM_LOCALID))
HOST=$(hostname -s)
echo "[$(date +%T)] task ${SLURM_PROCID} on ${HOST} localid=${SLURM_LOCALID} port=${PORT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

vllm serve '"${MODEL}"' \
    --host 0.0.0.0 \
    --port ${PORT} \
    --served-model-name Qwen/Qwen3.6-27B-xyixuan \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --reasoning-parser qwen3 \
    --trust-remote-code &

VLLM_PID=$!

# Wait for the endpoint to become healthy, then announce
while ! curl --noproxy "*" -s -o /dev/null -w "%{http_code}" http://${HOST}:${PORT}/health 2>/dev/null | grep -q "200"; do
    sleep 5
done

echo "${HOST}:${PORT}" > "'"${ENDPOINTS_DIR}"'/${SLURM_PROCID}.endpoint"
echo "[$(date +%T)] task ${SLURM_PROCID} READY at ${HOST}:${PORT}"

wait $VLLM_PID
'
