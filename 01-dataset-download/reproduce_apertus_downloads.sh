# Script that contains the commands called to download the Multimodal datasets for apertus 1.5 training

export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache"
export CACHE_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache"
export MEDICAL_DIR="/capstor/store/cscs/swissai/infra01/medical/raw"
export NUM_PROC=32
export MAX_RETRIES=15

# 1) Download Paired Img/Txt data - https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M
DATASET_NAME="mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M" sbatch 01-dataset-download/download_hf_dataset.slurm

# 2) Download Medical datasets
# a. HF well-formed datasets
DATASET_NAME="OpenMeditron/Mediset" sbatch 01-dataset-download/download_hf_dataset.slurm
DATASET_NAME="randall-lab/medmnist" TRUST_REMOTE_CODE="true" sbatch 01-dataset-download/download_hf_dataset.slurm
DATASET_NAME="allenai/pixmo-ask-model-anything" sbatch 01-dataset-download/download_hf_dataset.slurm
#
# b. HF ill-formed datasets
mkdir -p $MEDICAL_DIR/raw
hf download cyd0806/BIMCV-R --repo-type dataset --local-dir $MEDICAL_DIR/raw/BIMCV-R
hf download xmcmic/PMC-VQA --repo-type dataset --local-dir $MEDICAL_DIR/raw/PMC-VQA
#
# c. Non-HF datasets
zenodo_get https://doi.org/10.5281/zenodo.15814064 -o $MEDICAL_DIR/raw/MultiCaRe
#
# d. Datasets with manual download (due to ToS, licensing, etc)
#   - MURA (MSK Xrays):                    https://stanfordaimi.azurewebsites.net/datasets/3e00d84b-d86e-4fed-b2a4-bfe3effd661b
#   - MRNet (Knee MRI's):                  https://stanfordaimi.azurewebsites.net/datasets/bface6fc-7859-47d7-a1c8-022cd6b17419
#   - LERA (Lower Extremity Radiographs):  https://stanfordaimi.azurewebsites.net/datasets/44a63ddf-edd8-46cb-b021-094bb8efb802
#   - CoMM Dataset:                        https://github.com/HKUST-LongGroup/CoMM
#
# e. Datasets shared by the Meditron team (to be published on HF later)
#   - BUSI
#   - COVID_US
#   - DDTI
#   - PMC_VQA
#   - ct2
#   - image_mammoth
#   - iu_xray
#   - llava_instruct
#   - llava_pretrain_cleaned
#   - medtrinity_conversations_1_formatted
#   - medtrinity_conversations_2_formatted
#   - pixmo_anything
#   - pixmo_cap
