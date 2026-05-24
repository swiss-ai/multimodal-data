#!/usr/bin/env bash
# Downloads medical datasets from Zenodo, Figshare, and Kaggle.
#
# Prerequisites:
#   - wget
#   - Kaggle API key at ~/.kaggle/kaggle.json
#     (create at https://www.kaggle.com/settings → API → Create New Token)
#   - pip install kaggle
#
# Usage:
#   OUTPUT_DIR=/path/to/medical bash download.sh
#
# Individual sections can be commented out if the dataset is not needed.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/data/medical}"
mkdir -p "$OUTPUT_DIR"

# ── MedPix 2.0 ────────────────────────────────────────────────────────────────
# License: Creative Commons — see https://zenodo.org/records/12624810
mkdir -p "$OUTPUT_DIR/MedPix-2"
wget -c --timeout=60 --waitretry=10 --tries=5 \
    -O "${OUTPUT_DIR}/MedPix-2/MedPix-2_0.zip" \
    "https://zenodo.org/api/records/12624810/files-archive"

# ── NCT-CRC-HE-100K (colorectal cancer histology) ─────────────────────────────
# License: CC BY 4.0 — see https://zenodo.org/records/1214456
mkdir -p "$OUTPUT_DIR/NCT-CRC-HE-100K"
wget -c --timeout=60 --waitretry=10 --tries=5 \
    -O "${OUTPUT_DIR}/NCT-CRC-HE-100K/NCT-CRC-HE-100K.zip" \
    "https://zenodo.org/api/records/1214456/files-archive"

# ── VQA-Med (medical VQA) ──────────────────────────────────────────────────────
# Source: https://figshare.com/articles/dataset/VQA_Med_2019/9823281
mkdir -p "$OUTPUT_DIR/VQA-Med"
wget -c --timeout=60 --waitretry=10 --tries=5 \
    -O "${OUTPUT_DIR}/VQA-Med/VQA-Med.zip" \
    "https://figshare.com/ndownloader/files/3698839"

# ── Diabetic Retinopathy 2015 (Kaggle) ────────────────────────────────────────
# Requires Kaggle API key.
# Dataset page: https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized
kaggle datasets download \
    --path "${OUTPUT_DIR}/diabetic-retinopathy" \
    sovitrath/diabetic-retinopathy-2015-data-colored-resized

# ── Brain Tumor MRI (Kaggle) ──────────────────────────────────────────────────
# Requires Kaggle API key.
# Dataset page: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
kaggle datasets download \
    --path "${OUTPUT_DIR}/brain-tumor-mri" \
    masoudnickparvar/brain-tumor-mri-dataset

# ── Annotated Liver Ultrasound (Kaggle) ───────────────────────────────────────
# Requires Kaggle API key.
# Dataset page: https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset
kaggle datasets download \
    --path "${OUTPUT_DIR}/liver-ultrasound" \
    orvile/annotated-ultrasound-liver-images-dataset

# ── Breast Ultrasound (Kaggle) ────────────────────────────────────────────────
# Requires Kaggle API key.
# Dataset page: https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset
kaggle datasets download \
    --path "${OUTPUT_DIR}/breast-ultrasound" \
    sabahesaraki/breast-ultrasound-images-dataset

# ── DDTI Thyroid Ultrasound (Kaggle) ─────────────────────────────────────────
# Requires Kaggle API key.
# Dataset page: https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images
kaggle datasets download \
    --path "${OUTPUT_DIR}/ddti" \
    dasmehdixtr/ddti-thyroid-ultrasound-images

# ── COVID-19 Radiography Database (Kaggle) ────────────────────────────────────
# Requires Kaggle API key.
# Dataset page: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
kaggle datasets download \
    --path "${OUTPUT_DIR}/covid19-radiography" \
    tawsifurrahman/covid19-radiography-database

# ── NIH Chest X-rays (Kaggle) ─────────────────────────────────────────────────
# Requires Kaggle API key.
# Dataset page: https://www.kaggle.com/datasets/nih-chest-xrays/data
kaggle datasets download \
    --path "${OUTPUT_DIR}/nih-chest-xrays" \
    nih-chest-xrays/data

echo "Downloads complete. Unzip each archive in its subdirectory."
