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
hf download --repo-type dataset --revision 906a8b85cea61198ff7339c4dd711ad0b5361847 mlfoundations/MINT-1T-HTML
hf download --repo-type dataset --revision 7c5b00ffd5b563071010c3bf2082b4a8f836eb72 mlfoundations/MINT-1T-ArXiv
hf download --repo-type dataset --revision 089d3c500aded3a66f84d2ba05bddfd58e5ac8cc mlfoundations/MINT-1T-PDF-CC-2024-18
hf download --repo-type dataset --revision 4caa665264020fbe4a7b1fbca177445ca5897772 mlfoundations/MINT-1T-PDF-CC-2024-10
hf download --repo-type dataset --revision b9a0d67f6048cf79615e63fa44d7ac729958fc71 mlfoundations/MINT-1T-PDF-CC-2023-50
hf download --repo-type dataset --revision 3bc92bda919c7f05afd5f83af2e2f5ae9042eacd mlfoundations/MINT-1T-PDF-CC-2023-40
hf download --repo-type dataset --revision d2475fc14efd472e8c1cd10d1c0147b5fc52a2bf mlfoundations/MINT-1T-PDF-CC-2023-23
hf download --repo-type dataset --revision 6e5179c68a1dbf999fa29bf4665f763f8c62bb32 mlfoundations/MINT-1T-PDF-CC-2023-14
hf download --repo-type dataset --revision 2d9ed806777c02b5f6aaa25ec86250df3efe5ef5 mlfoundations/MINT-1T-PDF-CC-2023-06

# --- LAION Aesthetics 12M ---
# The HF dataset only contains URLs; actual images must be downloaded with img2dataset.
# See special/laion_aesthetics/ for the two-step download procedure.
hf download --repo-type dataset --revision 06928317703bcfa6099c7fc0f13e11bb295e7769 dclure/laion-aesthetics-12m-umap

# --- TreeOfLife-10M ---
# Requires license-based filtering before use (CC-BY / CC-0 only).
# See special/tree_of_life/ for the filtering script.
hf download --repo-type dataset --revision 91debffb7146c32c89d76feb1eb575b555e2ecc7 imageomics/TreeOfLife-10M

# --- BigDocs-7.5M ---
# The HF dataset must be supplemented with external image archives (COCO, TextVQA, TableFact).
# See special/bigdocs/ for the additional downloads.
hf download --repo-type dataset --revision dae4403c28307bd5328920740e81ce5232819e74 ServiceNow/BigDocs-7.5M
hf download --repo-type dataset --revision 35b1c097807e0b07ec5313879b85956b7b3890db bsmock/pubtables-1m

# --- Copernicus-Bench ---
# Satellite imagery benchmark. Raw files are GeoTIFFs; preprocessing converts them to RGB PNGs.
# See 02-data-preprocessing/ for the conversion step.
hf download --repo-type dataset --revision a287ab1b414d2bff99557166988571c5885ed81a wangyi111/Copernicus-Bench

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
hf download --repo-type dataset google/scin --revision 996257142f7517fb8991a28cfba46ec4e3f530a9 \
  --local-dir "$DATA_ROOT/medical/scin"

# --- PMC-OA (figures from open-access PubMed articles) ---
hf download --repo-type dataset --revision 97e6b8285d143c98a48b6a38ae13eccbf19c9738 axiong/pmc_oa

# --- OpenMed-PMC (open PMC 18M) ---
hf download --repo-type dataset vector-institute/open-pmc-18m --revision b5a67783ec3e1bf91809a5efc4b72fbedacacdf6 \
  --local-dir "$DATA_ROOT/medical/open-pmc-18m"

# --- MedTrinity-25M ---
hf download --repo-type dataset --revision 89e5c684794e5c4cc1af9e8f1a7798af7c937dbf UCSC-VLAA/MedTrinity-25M

# --- RFMiD2 (retinal fundus multi-disease image dataset) ---
# Available on Kaggle: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized
# Download with: kaggle datasets download sovitrath/diabetic-retinopathy-2015-data-colored-resized
# See special/medical/ for all Kaggle / Zenodo / Figshare medical downloads.

# --- UWF (ultra-widefield fundus) ---
# See special/medical/ — institutional dataset, not publicly redistributable.

# --- BUSI / COVIDx-US / DDTI ---
# These datasets were shared by the Meditron team.
# Original sources:
#   BUSI:      https://scholar.cu.edu.eg/?q=afahmy/pages/dataset
#   COVIDx-US: https://github.com/nrc-cnrc/COVID-US
#   DDTI:      https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images
# See special/medical/ for download instructions.

# --- HoloAssist (egocentric procedural video) ---
hf download --repo-type dataset --revision c18fb9c2c0b534193f5a5987fa9a9180505f59c2 mistakeattribution/holoassist

# --- SWISSIMAGE (Swiss aerial orthophoto) ---
# Downloaded from the swisstopo WMS API. Requires generating a URL list first.
# See special/swissimage/ for the full procedure.

# ============================================================
# STAGE 2 — Image + text pretraining
# ============================================================

# --- Recap-DataComp-1B (recaptioned subset) ---
hf download --repo-type dataset --revision 457c44d98651bcfdfb3cc8695f5e60a0d2705e78 UCSC-VLAA/Recap-DataComp-1B \
  --local-dir "$DATA_ROOT/Recap-DataComp-1B"

# --- LLaVA-OneVision Mid-Training 85M ---
hf download --repo-type dataset --revision c5218cad785eba7d218137e8ce4997bda568a050 mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M

# --- CommonCatalog CC-BY ---
hf download --repo-type dataset --revision 80f50fe4a1ca937f37a11be3f8eee5199d776ff3 common-canvas/commoncatalog-cc-by \
  --local-dir "$DATA_ROOT/commoncatalog-cc-by"

# --- PD12M (Spawning, full) ---
hf download --repo-type dataset --revision 7bda6bfe13c6a39fd0a286adced5df1228041720 Spawning/pd12m-full \
  --local-dir "$DATA_ROOT/pd12m-full"

# --- BLIP3 Grounding 50M ---
hf download --repo-type dataset --revision 4d622c4f19b8a6b91ad914caf343306e363ff79b Salesforce/blip3-grounding-50m \
  --local-dir "$DATA_ROOT/blip3-grounding-50m"

# --- PIN-200M ---
hf download --repo-type dataset --revision 0518ee384f3afbbc594f68c428667b86847327e8 m-a-p/PIN-200M

# --- BLIP3o Long Caption ---
hf download --repo-type dataset --revision e4d07091a466d1a1e35a9b0c61caddc78d14a059 BLIP3o/BLIP3o-Pretrain-Long-Caption

# --- MIT-10M ---
hf download --repo-type dataset --revision bcba6b2651771c69f93e000486c2baa0896d32c3 liboaccn/MIT-10M \
  --local-dir "$DATA_ROOT/MIT-10M"

# --- Art Museums Public Domain 440K ---
hf download --repo-type dataset --revision fba945da78b36262eb9272067197cc28d06cffbf Mitsua/art-museums-pd-440k \
  --local-dir "$DATA_ROOT/art-museums-pd-440k"

# --- Megalith-10M ---
hf download --repo-type dataset --revision 32000d81f4cf138e2ebc38163ec22998ff36ee6e madebyollin/megalith-10m \
  --local-dir "$DATA_ROOT/megalith-10m"

# --- UNO-1M ---
hf download --repo-type dataset --revision f25bb61db6d6d66d82f41d1e613c0e04ba342e84 bytedance-research/UNO-1M \
  --local-dir "$DATA_ROOT/UNO-1M"

# --- FaceCaption-15M ---
hf download --repo-type dataset --revision 3ed92d90f7fc7199b47c4da17c6863b1a175f380 OpenFace-CQUPT/FaceCaption-15M

# --- PixMo-Cap ---
# The HF dataset contains captions and image URLs; images must be downloaded with img2dataset.
# See special/pixmo_cap/ for the download procedure.
hf download --repo-type dataset --revision edce6390d9d5be6c8db0d863fbe62718c88988a4 allenai/pixmo-cap \
  --local-dir "$DATA_ROOT/pixmo-cap-meta"

# --- Molmo2 Synthetic Multi-Image QA ---
hf download --repo-type dataset --revision 5a00b7a26f0ad416ae7dd030094270e0a3adfb0f allenai/Molmo2-SynMultiImageQA

# --- SkyScript (satellite imagery + captions) ---
# Direct download from AWS S3. See special/skyscript/.
# https://github.com/wangzhecheng/SkyScript

# --- TCM Pretrain (web content) ---
hf download --repo-type dataset --revision db4874ce4e322f47432fe322c558e516c5aad71e FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT

# --- Fine-tuned T2I (curated subset) ---
hf download --repo-type dataset --revision 28fdd5663ee202b5cafc01d6ed08a03f14957854 ma-xu/fine-t2i --include "curated/**"

# --- MedMax (mint-medmax) ---
# See special/medmax/.

# --- RSTeller (remote-sensing captions) ---
hf download --repo-type dataset --revision a03b35f1bc9a3ac14ae93724d175c2611f1bba5b SlytherinGe/RSTeller

# --- MapTrace ---
hf download --repo-type dataset --revision 00bae0d2d917fd12548a089285d633dadf1bc81c google/MapTrace

# --- PMC-OA (Arxiv) ---
hf download --repo-type dataset --revision 1d2296e9c022a24e82a47e524d53f0915b98c926 axiong/pmc_oa

# --- GeoChat Instruct ---
hf download --repo-type dataset --revision 8eb13307eabc7fa9c1f8b0e61e372a327ccd68b1 MBZUAI/GeoChat_Instruct

# --- NIH Chest X-ray ---
# Available on Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
# Download with: kaggle datasets download nih-chest-xrays/data
# See special/medical/.

# --- DaTikZ v4 (TikZ figure–code pairs) ---
hf download --repo-type dataset --revision 33734c83608211682be11001a1618856fc1979dd nllg/DaTikZ-V4

# --- FLAIR-HUB (French aerial RGB) ---
hf download --repo-type dataset --revision 4cf55f57fd468fbd802681687c529a98c1274ce1 --include "data/*AERIAL_RGB*" IGNF/FLAIR-HUB

# --- Shopify Product Catalogue ---
hf download --repo-type dataset --revision d5c517c509f5aca99053897ef1de797d6d7e5aa5 Shopify/product-catalogue

# --- MultiCare (OpenMed) ---
hf download --repo-type dataset --revision 5c954c4fbf9abdcb55053488dab6c1ef142796b5 openmed-community/multicare-images
hf download --repo-type dataset --revision c8517124928d2fe3651ee6cb6c560fce66e02344 openmed-community/multicare-case-images

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
hf download --repo-type dataset --revision 9f20d0b0f6f48022bc2e10c46f219e3b89c44681 ai4ce/EgoPAT3Dv2

# ============================================================
# SFT — Supervised fine-tuning data
# ============================================================

# --- Innovator VL Instruct 46M ---
hf download --repo-type dataset --revision 1e7bac4db314fcc99e43601632c6ad7de71b3a6c InnovatorLab/Innovator-VL-Instruct-46M

# --- LLaVA-OneVision Instruct Data ---
hf download --repo-type dataset --revision d2d7906dc05edf1ba0e177be56b5b3c5c2c68608 mvp-lab/LLaVA-OneVision-1.5-Instruct-Data

# --- SenseNova SI 8M ---
hf download --repo-type dataset --revision 2f1c0b6136417f5e2423aff839086636858de3f0 sensenova/SenseNova-SI-8M

# --- Nemotron Image Training v3 ---
hf download --repo-type dataset --revision 7656391d4d4cb11ec3722b34f10d499435de0460 nvidia/Nemotron-Image-Training-v3 \
  --local-dir "$DATA_ROOT/sft/Nemotron-Image-Training-v3"

# --- MapTrace (SFT) ---
hf download --repo-type dataset --revision 00bae0d2d917fd12548a089285d633dadf1bc81c google/MapTrace

# --- RSRCC ---
hf download --repo-type dataset --revision 394d14dc9c171fbbbd03e02898c4e23e1bdfa657 google/RSRCC \
  --local-dir "$DATA_ROOT/sft/RSRCC"

# --- PixMo Cap QA ---
hf download --repo-type dataset --revision fd3ff4b2905455ab5edfd6d06039ea6027948521 allenai/pixmo-cap-qa \
  --local-dir "$DATA_ROOT/sft/pixmo-cap-qa"

# --- PixMo Ask Model Anything ---
hf download --repo-type dataset --revision 739ffd8b417737da31e5069b9c8ff6a72a7d58b6 allenai/pixmo-ask-model-anything \
  --local-dir "$DATA_ROOT/sft/pixmo-ask-model-anything"

# --- BigEarthNet v2 ---
# Hosted on Zenodo: https://zenodo.org/records/10891137
wget -c --timeout=60 --waitretry=10 --tries=5 \
  -P "$DATA_ROOT/BigEarthNet" \
  "https://zenodo.org/records/10891137/files/BigEarthNet-S1.tar.zst?download=1"
wget -c --timeout=60 --waitretry=10 --tries=5 \
  -P "$DATA_ROOT/BigEarthNet" \
  "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"

hf download --repo-type dataset --revision 3c380a731a3429c1d04693d6ec16d7e683def84c HuggingFaceM4/FineVision

# --- Molmo2 Multi-Image QA ---
hf download --repo-type dataset --revision f47ca3644d394b548be07a68d5a6fc0275924f08 allenai/Molmo2-MultiImageQA

# --- Molmo Point GUI Syn ---
hf download --repo-type dataset --revision 24bb3e990bb1e48796d80ade9dbc858dda695e75 allenai/MolmoPoint-GUISyn

# --- PathVQA ---
hf download --repo-type dataset --revision 1685832883334b5bb5beaf4e4b333fdeecaa4ad9 flaviagiammarino/path-vqa

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
