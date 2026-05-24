#!/usr/bin/env bash
# Downloads all datasets used in Apertus training.
#
# Prerequisites:
#   pip install huggingface_hub[cli] hf_transfer
#   huggingface-cli login   (or set HF_TOKEN)
#
# Configuration (set these before running):
#   HF_HUB_CACHE  - where to store HuggingFace Hub downloads
#                   (default: ~/.cache/huggingface/hub)
#   DATA_ROOT     - root directory for datasets that cannot go in the HF cache
#                   (non-HF downloads, img2dataset outputs, zenodo archives, etc.)
#
# Some datasets require special handling.
# They are noted inline with a reference to scripts in special/.

export HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"
export DATA_ROOT="${DATA_ROOT:-${HOME}/data}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$DATA_ROOT"

# ============================================================
# STAGE 1 — Image-only pretraining
# ============================================================

# --- MINT-1T ---
# Three sub-corpora: interleaved HTML pages, PDFs, and ArXiv papers.
hf download --repo-type dataset mlfoundations/MINT-1T-HTML
hf download --repo-type dataset mlfoundations/MINT-1T-ArXiv
hf download --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2024-18
hf download --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2024-10
hf download --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-50
hf download --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-40
hf download --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-23
hf download --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-14
hf download --repo-type dataset mlfoundations/MINT-1T-PDF-CC-2023-06

# --- LAION Aesthetics 12M ---
# The HF dataset only contains URLs; actual images must be downloaded with img2dataset.
# See special/laion_aesthetics/ for the two-step download procedure.
hf download --repo-type dataset dclure/laion-aesthetics-12m-umap

# --- TreeOfLife-10M ---
# Requires license-based filtering before use (CC-BY / CC-0 only).
# See special/tree_of_life/ for the filtering script.
hf download --repo-type dataset imageomics/TreeOfLife-10M

# --- BigDocs-7.5M ---
# The HF dataset must be supplemented with external image archives (COCO, TextVQA, TableFact).
# See special/bigdocs/ for the additional downloads.
hf download --repo-type dataset ServiceNow/BigDocs-7.5M
hf download --repo-type dataset bsmock/pubtables-1m

# --- Copernicus-Bench ---
# Satellite imagery benchmark. Raw files are GeoTIFFs; preprocessing converts them to RGB PNGs.
# See 02-data-preprocessing/ for the conversion step.
hf download --repo-type dataset wangyi111/Copernicus-Bench

# --- MedMNIST ---
# See special/medmnist/ — downloaded from Zenodo via zenodo_get.
# zenodo record: 10519652  (224-px variants: *_224.npz)

# --- MedMax ---
# Hosted as split tar.gz archives on HuggingFace Hub resolve endpoint.
# See special/medmax/ for the wget-based download.

# --- SLIDE (histopathology) ---
# Hosted on Figshare.
# See special/slide/ for the wget-based download.

# --- SCIN (skin condition image network) ---
# Gated dataset: accept terms at https://huggingface.co/datasets/google/scin first.
hf download --repo-type dataset google/scin \
  --local-dir "$DATA_ROOT/medical/scin"

# --- PMC-OA (figures from open-access PubMed articles) ---
hf download --repo-type dataset axiong/pmc_oa

# --- OpenMed-PMC (open PMC 18M) ---
hf download --repo-type dataset vector-institute/open-pmc-18m \
  --local-dir "$DATA_ROOT/medical/open-pmc-18m"

# --- MedTrinity-25M ---
hf download --repo-type dataset UCSC-VLAA/MedTrinity-25M

# --- RFMiD2 (retinal fundus multi-disease image dataset) ---
# Available on Kaggle: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized
# Download with: kaggle datasets download sovitrath/diabetic-retinopathy-2015-data-colored-resized
# See special/medical/ for all Kaggle / Zenodo / Figshare medical downloads.

# --- UWF (ultra-widefield fundus) ---
# See special/medical/ — institutional dataset, not publicly redistributable.

# --- BUSI / COVIDx-US / DDTI ---
# These datasets were shared by the Meditron team.
# Original sources:
#   BUSI:     https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
#   COVIDx-US: https://github.com/nrc-cnrc/COVID-US
#   DDTI:     https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images
# See special/medical/ for download instructions.

# --- HoloAssist (egocentric procedural video) ---
hf download --repo-type dataset microsoft/HoloAssist

# --- SWISSIMAGE (Swiss aerial orthophoto) ---
# Downloaded from the swisstopo WMS API. Requires generating a URL list first.
# See special/swissimage/ for the full procedure.

# ============================================================
# STAGE 2 — Image + text pretraining
# ============================================================

# --- Recap-DataComp-1B (recaptioned subset) ---
hf download --repo-type dataset UCSC-VLAA/Recap-DataComp-1B \
  --local-dir "$DATA_ROOT/Recap-DataComp-1B"

# --- LLaVA-OneVision Mid-Training 85M ---
hf download --repo-type dataset mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M

# --- CommonCatalog CC-BY ---
hf download --repo-type dataset common-canvas/commoncatalog-cc-by \
  --local-dir "$DATA_ROOT/commoncatalog-cc-by"

# --- PD12M (Spawning, full) ---
hf download --repo-type dataset Spawning/pd12m-full \
  --local-dir "$DATA_ROOT/pd12m-full"

# --- BLIP3 Grounding 50M ---
hf download --repo-type dataset Salesforce/blip3-grounding-50m \
  --local-dir "$DATA_ROOT/blip3-grounding-50m"

# --- PIN-200M ---
hf download --repo-type dataset m-a-p/PIN-200M

# --- BLIP3o Long Caption ---
hf download --repo-type dataset BLIP3o/BLIP3o-Pretrain-Long-Caption

# --- MIT-10M ---
hf download --repo-type dataset liboaccn/MIT-10M \
  --local-dir "$DATA_ROOT/MIT-10M"

# --- Art Museums Public Domain 440K ---
hf download --repo-type dataset Mitsua/art-museums-pd-440k \
  --local-dir "$DATA_ROOT/art-museums-pd-440k"

# --- Megalith-10M ---
hf download --repo-type dataset madebyollin/megalith-10m \
  --local-dir "$DATA_ROOT/megalith-10m"

# --- WebSight V0.2 (webpage screenshots) ---
hf download --repo-type dataset HuggingFaceM4/WebSight

# --- OpenPMC (Vector Institute) ---
hf download --repo-type dataset vector-institute/open-pmc

# --- TextAtlas-5M ---
hf download --repo-type dataset CSU-JPG/TextAtlas5M

# --- Open Images V7 ---
hf download --repo-type dataset bitmind/open-images-v7

# --- MedTrinity-25M (paired) ---
hf download --repo-type dataset UCSC-VLAA/MedTrinity-25M

# --- UNO-1M ---
hf download --repo-type dataset bytedance-research/UNO-1M \
  --local-dir "$DATA_ROOT/UNO-1M"

# --- FaceCaption-15M ---
hf download --repo-type dataset OpenFace-CQUPT/FaceCaption-15M

# --- PixMo-Cap ---
# The HF dataset contains captions and image URLs; images must be downloaded with img2dataset.
# See special/pixmo_cap/ for the download procedure.
hf download --repo-type dataset allenai/pixmo-cap \
  --local-dir "$DATA_ROOT/pixmo-cap-meta"

# --- Molmo2 Synthetic Multi-Image QA ---
hf download --repo-type dataset allenai/Molmo2-SynMultiImageQA

# --- SkyScript (satellite imagery + captions) ---
# Direct download from AWS S3. See special/skyscript/.
# https://github.com/wangzhecheng/SkyScript

# --- TCM Pretrain (web content) ---
hf download --repo-type dataset FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT

# --- Fine-tuned T2I (curated subset) ---
hf download --repo-type dataset ma-xu/fine-t2i \
  --include "curated/**"

# --- MedMax (mint-medmax) ---
# See special/medmax/.

# --- RSTeller (remote-sensing captions) ---
hf download --repo-type dataset SlytherinGe/RSTeller

# --- MapTrace ---
hf download --repo-type dataset google/MapTrace

# --- PMC-OA (Arxiv) ---
hf download --repo-type dataset axiong/pmc_oa

# --- GeoChat Instruct ---
hf download --repo-type dataset MBZUAI/GeoChat_Instruct

# --- NIH Chest X-ray ---
# Available on Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
# Download with: kaggle datasets download nih-chest-xrays/data
# See special/medical/.

# --- DaTikZ v4 (TikZ figure–code pairs) ---
hf download --repo-type dataset nllg/DaTikZ-V4

# --- FLAIR-HUB (French aerial RGB) ---
hf download --repo-type dataset --include "data/*AERIAL_RGB*" IGNF/FLAIR-HUB

# --- Shopify Product Catalogue ---
hf download --repo-type dataset Shopify/product-catalogue

# --- MultiCare (OpenMed) ---
hf download --repo-type dataset openmed-community/multicare-images
hf download --repo-type dataset openmed-community/multicare-case-images

# --- NCT-CRC-HE-100K (colorectal histology patches) ---
# Hosted on Zenodo: https://zenodo.org/records/1214456
# Download with: wget https://zenodo.org/api/records/1214456/files-archive -O NCT-CRC-HE-100K.zip
# See special/medical/.

# --- MedPix-2 ---
# Hosted on Zenodo: https://zenodo.org/records/12624810
# Download with: wget https://zenodo.org/api/records/12624810/files-archive -O MedPix-2_0.zip
# See special/medical/.

# --- EBHI-Seg (endoscopic biopsy histology) ---
# Available via Dataset Ninja: https://datasetninja.com/ebhi-seg
# See special/medical/.

# --- Diabetic Retinopathy 2015 (Kaggle) ---
# Available on Kaggle: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized
# Download with: kaggle datasets download sovitrath/diabetic-retinopathy-2015-data-colored-resized
# See special/medical/.

# --- Brain Tumor MRI (Kaggle) ---
# Available on Kaggle: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# Download with: kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset
# See special/medical/.

# --- Liver Ultrasound (Kaggle) ---
# Available on Kaggle: https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset
# Download with: kaggle datasets download orvile/annotated-ultrasound-liver-images-dataset
# See special/medical/.

# --- Breast Ultrasound (Kaggle) ---
# Available on Kaggle: https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset
# Download with: kaggle datasets download sabahesaraki/breast-ultrasound-images-dataset
# See special/medical/.

# --- EgoPAT3D v2 ---
hf download --repo-type dataset ai4ce/EgoPAT3Dv2

# ============================================================
# SFT — Supervised fine-tuning data
# ============================================================

# --- Innovator VL Instruct 46M ---
hf download --repo-type dataset InnovatorLab/Innovator-VL-Instruct-46M

# --- LLaVA-OneVision Instruct Data ---
hf download --repo-type dataset mvp-lab/LLaVA-OneVision-1.5-Instruct-Data

# --- SenseNova SI 8M ---
hf download --repo-type dataset sensenova/SenseNova-SI-8M

# --- Nemotron Image Training v3 ---
hf download --repo-type dataset nvidia/Nemotron-Image-Training-v3 \
  --local-dir "$DATA_ROOT/sft/Nemotron-Image-Training-v3"

# --- MapTrace (SFT) ---
hf download --repo-type dataset google/MapTrace

# --- RSRCC ---
hf download --repo-type dataset google/RSRCC \
  --local-dir "$DATA_ROOT/sft/RSRCC"

# --- PixMo Cap QA ---
hf download --repo-type dataset allenai/pixmo-cap-qa \
  --local-dir "$DATA_ROOT/sft/pixmo-cap-qa"

# --- PixMo Ask Model Anything ---
hf download --repo-type dataset allenai/pixmo-ask-model-anything \
  --local-dir "$DATA_ROOT/sft/pixmo-ask-model-anything"

# --- BigEarthNet v2 ---
# Hosted on Zenodo: https://zenodo.org/records/10891137
wget -c --timeout=60 --waitretry=10 --tries=5 \
  -P "$DATA_ROOT/BigEarthNet" \
  "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1"
wget -c --timeout=60 --waitretry=10 --tries=5 \
  -P "$DATA_ROOT/BigEarthNet" \
  "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"

# --- CoSyn 400K ---
hf download --repo-type dataset finevision/CoSyn_400k

# --- Molmo2 Multi-Image QA ---
hf download --repo-type dataset allenai/Molmo2-MultiImageQA

# --- Molmo Point GUI Syn ---
hf download --repo-type dataset allenai/MolmoPoint-GUISyn

# --- PathVQA ---
hf download --repo-type dataset flaviagiammarino/path-vqa

# --- MultiHierTT ---
hf download --repo-type dataset finevision/multihiertt

# --- TCM Instruction Tuning ---
hf download --repo-type dataset FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT

# ============================================================
# Notes on datasets not listed above
# ============================================================
#
# SWISSIMAGE, LAION-Aesthetics-12M, SkyScript, MedMax, MedMNIST, SLIDE,
# PixMo-Cap images, and several medical datasets (BUSI, COVIDx-US, DDTI,
# UWF fundus, MedPix-2, NCT-CRC-HE-100K, EBHI-Seg, etc.) require
# dataset-specific download procedures.
# See the special/ subdirectory for individual download scripts.
#
# The PMC-OA bulk text files (oa_comm / oa_other / oa_noncomm) can be
# synced from the NCBI FTP server:
#   rclone sync :http,url=https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/ "$DATA_ROOT/pmc_oa_bulk" \
#     --include "/oa_comm/txt/**" --include "/oa_other/txt/**" --include "/oa_noncomm/txt/**"
