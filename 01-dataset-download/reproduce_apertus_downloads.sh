# Script that contains the commands called to download the Multimodal datasets for apertus 1.5 training

export HF_HUB_CACHE="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache"
export CACHE_DIR="/capstor/store/cscs/swissai/infra01/vision-datasets/hf_datasets_cache"
export MEDICAL_DIR="/capstor/store/cscs/swissai/infra01/medical/raw"
export NUM_PROC=32
export MAX_RETRIES=15

export VISION_DATASETS_ROOT="/capstor/store/cscs/swissai/infra01/vision-datasets"
export HF_TOKEN="$(cat ${HOME}/.cache/huggingface/token)"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=1

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

# -- MINT-1T
hf download --repo-type dataset --revision 906a8b85cea61198ff7339c4dd711ad0b5361847 mlfoundations/MINT-1T-HTML
hf download --repo-type dataset --revision 7c5b00ffd5b563071010c3bf2082b4a8f836eb72 mlfoundations/MINT-1T-ArXiv
hf download --repo-type dataset --revision 089d3c500aded3a66f84d2ba05bddfd58e5ac8cc mlfoundations/MINT-1T-PDF-CC-2024-18
hf download --repo-type dataset --revision 4caa665264020fbe4a7b1fbca177445ca5897772 mlfoundations/MINT-1T-PDF-CC-2024-10
hf download --repo-type dataset --revision b9a0d67f6048cf79615e63fa44d7ac729958fc71 mlfoundations/MINT-1T-PDF-CC-2023-50
hf download --repo-type dataset --revision 3bc92bda919c7f05afd5f83af2e2f5ae9042eacd mlfoundations/MINT-1T-PDF-CC-2023-40
hf download --repo-type dataset --revision d2475fc14efd472e8c1cd10d1c0147b5fc52a2bf mlfoundations/MINT-1T-PDF-CC-2023-23
hf download --repo-type dataset --revision 6e5179c68a1dbf999fa29bf4665f763f8c62bb32 mlfoundations/MINT-1T-PDF-CC-2023-14
hf download --repo-type dataset --revision 2d9ed806777c02b5f6aaa25ec86250df3efe5ef5 mlfoundations/MINT-1T-PDF-CC-2023-06

# -- LAION Aesthetics 12M
hf download --repo-type dataset --revision 06928317703bcfa6099c7fc0f13e11bb295e7769 dclure/laion-aesthetics-12m-umap

# -- TreeOfLife 10M
hf download --repo-type dataset --revision 91debffb7146c32c89d76feb1eb575b555e2ecc7 imageomics/TreeOfLife-10M

# -- Latex Formulas 80M
hf download --repo-type dataset --revision 5cd783320b0092caa85720a85d86595f42df043b OleehyO/latex-formulas-80M

# -- BigDocs-7.5M
hf download --repo-type dataset --revision dae4403c28307bd5328920740e81ce5232819e74 ServiceNow/BigDocs-7.5M
hf download --repo-type dataset --revision 35b1c097807e0b07ec5313879b85956b7b3890db bsmock/pubtables-1m
cat <<EOF | xargs -n 1 -P 3 wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/bigDocs7_5m/subsets
http://images.cocodataset.org/zips/train2014.zip
https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
https://tablefact.s3-us-west-2.amazonaws.com/preprocessed_data_program.zip
EOF

# -- YODAS2
hf download --repo-type dataset --revision c9674490249665d658f527e2684848377108d82c --local-dir /iopsstor/scratch/cscs/tchu/shared/yodas2 espnet/yodas2

# -- Geospatial Imagery
# Copernicus-Bench
hf download --repo-type dataset --revision a287ab1b414d2bff99557166988571c5885ed81a wangyi111/Copernicus-Bench
# BigEarthNet
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/BigEarthNet https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/BigEarthNet https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1
# MapTrace
hf download --repo-type dataset --revision 00bae0d2d917fd12548a089285d633dadf1bc81c google/MapTrace
# GeoChat
hf download --repo-type dataset --revision 8eb13307eabc7fa9c1f8b0e61e372a327ccd68b1 MBZUAI/GeoChat_Instruct
# RSTeller
hf download --repo-type dataset --revision a03b35f1bc9a3ac14ae93724d175c2611f1bba5b SlytherinGe/RSTeller
# FLAIR-HUB (Aerial RGB)
hf download --repo-type dataset --revision 4cf55f57fd468fbd802681687c529a98c1274ce1 --include "data/*AERIAL_RGB*" IGNF/FLAIR-HUB
# SkyScript
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/images2.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/images3.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/images4.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/images5.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/images6.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/images7.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/meta2.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/meta3.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/meta4.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/meta5.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/meta6.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/meta7.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -P $VISION_DATASETS_ROOT/wangzhecheng https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/SkyScript_train_unfiltered_5M.csv

# -- Medical Apertus Expansion
hf download --repo-type dataset --revision 3c84e04b38bceb5341419b9a4f8ca37ba790cb84 FreedomIntelligence/PubMedVision
hf download --repo-type dataset --revision bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9 flaviagiammarino/vqa-rad
hf download --repo-type dataset --revision 1685832883334b5bb5beaf4e4b333fdeecaa4ad9 flaviagiammarino/path-vqa
hf download --repo-type dataset --revision 794df7cf2fe83d6ac43c78a591fda52bc67ec5e0 Voxel51/SLAKE
hf download --repo-type dataset --revision 142f64e5a8e8084301562f81034698f21a325004 faizan711/VinDR-CXR-VQA
hf download --repo-type dataset --revision 3e94599d7e41d9992a8362cc0467eb397e11210f bumbledeep/eyepacs
hf download --repo-type dataset --revision 5c954c4fbf9abdcb55053488dab6c1ef142796b5 openmed-community/multicare-images
hf download --repo-type dataset --revision c8517124928d2fe3651ee6cb6c560fce66e02344 openmed-community/multicare-case-images
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/MedPix-2_0.zip https://zenodo.org/api/records/12624810/files-archive
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/VQA-Med.zip https://figshare.com/ndownloader/files/3698839
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/covid19-radiography-database.zip https://www.kaggle.com/api/v1/datasets/download/tawsifurrahman/covid19-radiography-database
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/nih-chest-xrays.zip https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/Mendeley_Digital_Knee_X-ray.zip https://data.mendeley.com/public-api/zip/t9ndx37v5h/download/1
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/brain-tumor-mri-dataset.zip https://www.kaggle.com/api/v1/datasets/download/masoudnickparvar/brain-tumor-mri-dataset
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/abdominal-ultrasound-images.zip https://www.kaggle.com/api/v1/datasets/download/darsh22blc1378/abdominal-ultrasound-images
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/annotated-ultrasound-liver-images-dataset.zip https://www.kaggle.com/api/v1/datasets/download/orvile/annotated-ultrasound-liver-images-dataset
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/notop-wg-radimagenet.zip https://www.kaggle.com/api/v1/datasets/download/ipythonx/notop-wg-radimagenet
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/PAPILA.zip https://figshare.com/ndownloader/files/35013982
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/diabetic-retinopathy-2015-data-colored-resized.zip https://www.kaggle.com/api/v1/datasets/download/sovitrath/diabetic-retinopathy-2015-data-colored-resized
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/breast-cancer-wisconsin-diagnostic.zip https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/EBHI-Seg.tar https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMzMyOV9FQkhJLVNlZy9lYmhpc2VnLURhdGFzZXROaW5qYS50YXIiLCAic2lnIjogIjhocUpGcTBIVkNlV1FVS3NjRHVqenNHeGR1aUxQR2ZrMGduQ3owWlBXSFU9In0=?response-content-disposition=attachment%3B%20filename%3D%22ebhiseg-DatasetNinja.tar%22
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/BCCD.zip https://github.com/Shenggan/BCCD_Dataset/releases/download/v1.0/bccd_rec.zip
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/BioMediTech_RPE_dataset.zip https://figshare.com/ndownloader/articles/2070109/versions/1
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/medicat.tar.gz https://ai2-s2-medicat.s3.us-west-2.amazonaws.com/2020-10-05/medicat_release.tar.gz
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/NCT-CRC-HE-100K.zip https://zenodo.org/api/records/1214456/files-archive
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/CoNSeP.zip https://www.kaggle.com/api/v1/datasets/download/karthikperupogu/consep
wget -c --timeout=60 --waitretry=10 --tries=5 -O $VISION_DATASETS_ROOT/medical/raw/apertus/Gleason-TMA.zip "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/OCYCMP"
gdown --folder https://drive.google.com/drive/folders/1gHou49cA1s5vua2V5L98Lt8TiWA3FrKB -O $VISION_DATASETS_ROOT/medical/raw/apertus/PatchCamelyon

# -- UNO-1M and MIT-10M
hf download --repo-type dataset --revision f25bb61db6d6d66d82f41d1e613c0e04ba342e84 bytedance-research/UNO-1M
hf download --repo-type dataset --revision bcba6b2651771c69f93e000486c2baa0896d32c3 liboaccn/MIT-10M

# hf cache verify --repo-type dataset mlfoundations/MINT-1T-HTML
# hf cache verify --repo-type dataset mlfoundations/MINT-1T-ArXiv
# hf cache verify --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2024-18
# hf cache verify --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2024-10
# hf cache verify --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-50
# hf cache verify --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-40
# hf cache verify --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-23
# hf cache verify --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-14
# hf cache verify --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-06
# hf cache verify --repo-type dataset ServiceNow/BigDocs-7.5M
# hf cache verify --repo-type dataset dclure/laion-aesthetics-12m-umap
# hf cache verify --repo-type dataset FreedomIntelligence/PubMedVision
# hf cache verify --repo-type dataset flaviagiammarino/vqa-rad
# hf cache verify --repo-type dataset flaviagiammarino/path-vqa
# hf cache verify --repo-type dataset Voxel51/SLAKE
