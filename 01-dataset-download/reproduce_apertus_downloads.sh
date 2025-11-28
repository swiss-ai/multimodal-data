# Script that contains the commands called to download the Multimodal datasets for apertus 1.5 training

export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache"
export CACHE_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache"
export NUM_PROC=32
export MAX_RETRIES=15

# 1) Download Paired Img/Txt data - https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M
DATASET_NAME="mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M" sbatch 01-dataset-download/download_hf_dataset.slurm
