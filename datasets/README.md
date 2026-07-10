# Datasets

## Vision - Stage 1

### Image (general, satellite, bio, ocr)

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| TreeOfLife-10M | image | Stage 1 | [tree_of_life/](datasets/tree_of_life/) | [source](https://huggingface.co/datasets/imageomics/TreeOfLife-10M/tree/91debffb7146c32c89d76feb1eb575b555e2ecc7) |
| LAION Aesthetics 12M UMAP | image | Stage 1 | [laion/](datasets/laion/), [laion.py](medical/adapters/laion.py) | [source](https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap/tree/06928317703bcfa6099c7fc0f13e11bb295e7769) |
| MINT-1T HTML | image-text | Stage 1 | [mint_html/](datasets/mint_html/) | [source](https://huggingface.co/datasets/mlfoundations/MINT-1T-HTML/tree/906a8b85cea61198ff7339c4dd711ad0b5361847) |
| MINT-1T PDF | image | Stage 1 | [mint_arxiv/](datasets/mint_arxiv/) | [source](https://huggingface.co/collections/mlfoundations/mint-1t) |
| MINT-1T ArXiv | image | Stage 1 | [mint_arxiv/](datasets/mint_arxiv/) | [source](https://huggingface.co/datasets/mlfoundations/MINT-1T-ArXiv/tree/7c5b00ffd5b563071010c3bf2082b4a8f836eb72) |
| BigDocs-7.5M | document | Stage 1 | [bigdocs/](datasets/bigdocs/) | [source](https://huggingface.co/datasets/ServiceNow/BigDocs-7.5M/tree/dae4403c28307bd5328920740e81ce5232819e74) |
| SWISSIMAGE 10cm | image | Stage 1, Cooldown & LCP | [swissimage/](datasets/swissimage/), [swisstopo/](datasets/swisstopo/) | [source](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10#Download) |
| Copernicus-Bench (bigearthnet, dfc2020, eurosat) | image | Stage 1 | [copernicus/](datasets/copernicus/) | [source](https://huggingface.co/datasets/wangyi111/Copernicus-Bench/tree/a287ab1b414d2bff99557166988571c5885ed81a) |
| MMammoTH | image-text | Stage 1 | [preprocessing/](preprocessing/) | [source](https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M/tree/bac8f77cb8a8f9c4d0de407c6e3a589bd722562a) |
| HoloAssist | image | Stage 1 | [holoassist.py](medical/adapters/holoassist.py), [apertus1p5/](medical/configs/apertus1p5/) | [source](https://holoassist.github.io/) |
| Nicola Handwriting - Docs | image | Stage 1 | TODO | internal |
| Nicola Handwriting - Slides | image | Stage 1 | TODO | internal |

### Medical - Stage 1 (via `medical` pipeline)

|Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| Breast Ultrasound (BUSI) | image | Stage 1, Stage 2 | [preprocessing/](preprocessing/) | [source](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) |
| COVID-US | image | Stage 1, Stage 2 | [preprocessing/](preprocessing/) | [source](https://github.com/nrc-cnrc/COVID-US) |
| DDTI Thyroid Ultrasound | image | Stage 1, Stage 2 | [preprocessing/](preprocessing/) | [source](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) |
| ISIC Skin Disease | image | Stage 1 | [isic.py](medical/adapters/isic.py) | [source](https://www.kaggle.com/datasets/abhii1929/isic-skin-disease-image-dataset-4-classes) |
| MedMax | image-text | Stage 1, Stage 2 | [medmax.py](medical/adapters/medmax.py), [preprocessing/](preprocessing/) | [source](https://huggingface.co/datasets/mint-medmax/medmax_data/tree/2a63fb4d5e57bbf30a130976825fadf95ea95ebb) |
| MedMNIST | image | Stage 1 | [medmnist.py](medical/adapters/medmnist.py) | [source](https://zenodo.org/records/10519652) |
| MedTrinity-25M | image-text | Stage 1, Stage 2 | [medtrinity.py](medical/adapters/medtrinity.py), [preprocessing/](preprocessing/) | [source](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M/tree/89e5c684794e5c4cc1af9e8f1a7798af7c937dbf) |
| OpenPMC-18M | image | Stage 1, Stage 2 | [open_pmc_18m.py](medical/adapters/open_pmc_18m.py) | [source](https://huggingface.co/datasets/vector-institute/open-pmc-18m/tree/b5a67783ec3e1bf91809a5efc4b72fbedacacdf6) |
| PMC OA | image | Stage 1, Stage 2 | [pmc_oa.py](medical/adapters/pmc_oa.py), [preprocessing/](preprocessing/) | [source](https://huggingface.co/datasets/axiong/pmc_oa/tree/1d2296e9c022a24e82a47e524d53f0915b98c926) |
| RFMiD 2 | image | Stage 1 | [rfmid2.py](medical/adapters/rfmid2.py) | [source](https://zenodo.org/records/7505822) |
| SciN | image | Stage 1 | [scin.py](medical/adapters/scin.py) | [source](https://huggingface.co/datasets/google/scin/tree/996257142f7517fb8991a28cfba46ec4e3f530a9) |
| SLID-E | image | Stage 1 | [slide.py](medical/adapters/slide.py) | [source](https://figshare.com/articles/dataset/SLID-E/26172919) |
| Open UFI | image | Stage 1 | [uwf.py](medical/adapters/uwf.py) | [source](https://springernature.figshare.com/articles/dataset/Open_UFI_and_clinical_IQA/26936446) |

## Vision - Stage 2

### Image-to-text

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| Recap-DataComp-1B | image-text | Stage 2, Cooldown | [recap_datacomp/](datasets/recap_datacomp/), [recap_datacomp_1b/](datasets/recap_datacomp_1b/) | [source](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B/tree/457c44d98651bcfdfb3cc8695f5e60a0d2705e78) |
| LaTeX Formulas 80M | image-text | Stage 2, Cooldown | [latex_formulas_80m/](datasets/latex_formulas_80m/) | [source](https://huggingface.co/datasets/OleehyO/latex-formulas-80M/tree/5cd783320b0092caa85720a85d86595f42df043b) |
| BLIP3 Grounding 50M | image-text | Stage 2 | [blip3_grounding/](datasets/blip3_grounding/) | [source](https://huggingface.co/datasets/Salesforce/blip3-grounding-50m/tree/4d622c4f19b8a6b91ad914caf343306e363ff79b) |
| PD12M Full | image-text | Stage 2 | [pd12m/](datasets/pd12m/) | [source](https://huggingface.co/datasets/Spawning/pd12m-full/tree/7bda6bfe13c6a39fd0a286adced5df1228041720) |
| FaceCaption-15M | image-text | Stage 2 | [facecaption_15m/](datasets/facecaption_15m/), [facetaption/](datasets/facetaption/) | [source](https://huggingface.co/datasets/OpenFace-CQUPT/FaceCaption-15M/tree/3ed92d90f7fc7199b47c4da17c6863b1a175f380) |
| Megalith-10M | image-text | Stage 2 | [megalith/](datasets/megalith/) | [source](https://huggingface.co/datasets/madebyollin/megalith-10m/tree/1e65a79953396f6d05f60eba4cc564541ad4be8c) |
| SkyScript | image-text | Stage 2, Cooldown | [skyscript/](datasets/skyscript/) | [source](https://github.com/wangzhecheng/SkyScript) |
| WebSight V0.2 | image-text | Stage 2 | [websight/](datasets/websight/) | [source](https://huggingface.co/datasets/HuggingFaceM4/WebSight/tree/b11f8172f89c992b56ac702319e02c428cca4a4e) |
| Open Images V7 | image-text | Stage 2 | [openimages/](datasets/openimages/) | [source](https://huggingface.co/datasets/bitmind/open-images-v7/tree/4518ecd40f8f9ef66ee4356be438f840c714e95a) |
| DaTikZ-V4 | image-text | Stage 2 | [datikz_v4/](datasets/datikz_v4/) | [source](https://huggingface.co/datasets/nllg/DaTikZ-V4/tree/33734c83608211682be11001a1618856fc1979dd) |
| Art Museums PD 440k | image-text | Stage 2, Cooldown | [art_museums_pd/](datasets/art_museums_pd/) | [source](https://huggingface.co/datasets/Mitsua/art-museums-pd-440k/tree/fba945da78b36262eb9272067197cc28d06cffbf) |
| Fine-T2I | image-text | Stage 2 | [fine_t2i/](datasets/fine_t2i/) | [source](https://huggingface.co/datasets/ma-xu/fine-t2i/tree/28fdd5663ee202b5cafc01d6ed08a03f14957854) |
| MapTrace | image-text | Stage 2 | [maptrace/](datasets/maptrace/) | [source](https://huggingface.co/datasets/google/MapTrace/tree/8dd60adfde2f189768f27204c78ec44af07a67bf) |
| FLAIR-HUB | image | Stage 2 | [flair_hub/](datasets/flair_hub/), [ign/](datasets/ign/) | [source](https://huggingface.co/datasets/IGNF/FLAIR-HUB/tree/8275163f72f0eed69050a703925791b7c3577f10) |
| IGN City Tiles | image | Stage 2 | [ign/](datasets/ign/) | [source](https://geoservices.ign.fr/planign) |
| Shopify Product Catalogue | image-text | Stage 2 | [product_catalogue/](datasets/product_catalogue/) | [source](https://huggingface.co/datasets/Shopify/product-catalogue/tree/d5c517c509f5aca99053897ef1de797d6d7e5aa5) |
| EgoPAT3Dv2 | image | Stage 2, Cooldown | [egopat3dv2/](datasets/egopat3dv2/) | [source](https://huggingface.co/datasets/ai4ce/EgoPAT3Dv2/tree/9f20d0b0f6f48022bc2e10c46f219e3b89c44681) |
| LLaVA-OV Mid-Training 85M | image-text | Stage 2 | TODO | [source](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/tree/c5218cad785eba7d218137e8ce4997bda568a050) |
| BLIP3o Long Caption | image-text | Stage 2 | TODO | [source](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption/tree/e4d07091a466d1a1e35a9b0c61caddc78d14a059) |
| CommonCatalog CC-BY | image-text | Stage 2 | [loader_commoncatalog.py](recaption/vllm/loader_commoncatalog.py) | [source](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by/tree/80f50fe4a1ca937f37a11be3f8eee5199d776ff3) |
| TextAtlas5M | image-text | Stage 2 | TODO | [source](https://huggingface.co/datasets/CSU-JPG/TextAtlas5M/tree/f9f2a0f5000fbb078f718197acb45cfb9ceed551) |
| UNO-1M | image-text | Stage 2 | [loader_uno_1m_v3.py](recaption/vllm/loader_uno_1m_v3.py) | [source](https://huggingface.co/datasets/bytedance-research/UNO-1M/tree/f25bb61db6d6d66d82f41d1e613c0e04ba342e84) |
| RSTeller | image-text | Stage 2 | [rsteller.py](image_text/rsteller.py) | [source](https://huggingface.co/datasets/SlytherinGe/RSTeller/tree/a03b35f1bc9a3ac14ae93724d175c2611f1bba5b) |
| GeoChat Instruct | image-text | Stage 2 | [geochat.py](image_text/geochat.py) | [source](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct/tree/8eb13307eabc7fa9c1f8b0e61e372a327ccd68b1) |
| MIT-10M | image-text | Stage 2 | [loader_mit_10m_qwen_from_text.py](recaption/vllm/loader_mit_10m_qwen_from_text.py) | [source](https://huggingface.co/datasets/liboaccn/MIT-10M/tree/bcba6b2651771c69f93e000486c2baa0896d32c3) |
| PixMo-Cap | image-text | Stage 2, Cooldown | [pixmo/](datasets/pixmo/), [pixmo_cap/](download/special/pixmo_cap/), [preprocessing/](preprocessing/) | [source](https://huggingface.co/datasets/allenai/pixmo-cap/tree/edce6390d9d5be6c8db0d863fbe62718c88988a4) |

### Interleave

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| PIN-200M | image-text | Stage 2, Cooldown & LCP | [pin_200m/](datasets/pin_200m/) | [source](https://huggingface.co/datasets/m-a-p/PIN-200M/tree/f69c5da58f4284c6687a0e058c21e67fca9a1b66) |
| TCM Shizhen (Book Vision + Web Vision) | image-text | Stage 2, Cooldown & LCP | [tcm_pretrain_shizhen/](datasets/tcm_pretrain_shizhen/) | [source](https://huggingface.co/datasets/FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT/tree/db4874ce4e322f47432fe322c558e516c5aad71e) |
| Molmo2 Syn MultiImage | image-text | Stage 2, Cooldown | [molmo2/](datasets/molmo2/), [molmo2_synmultiimageqa/](datasets/molmo2_synmultiimageqa/) | [source](https://huggingface.co/datasets/allenai/Molmo2-MultiImageQA/tree/f47ca3644d394b548be07a68d5a6fc0275924f08) |

### Medical - Stage 2 (via `medical` pipeline)

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| Brain Tumor MRI | image | Stage 2 | [brain_tumor_mri.py](medical/adapters/brain_tumor_mri.py) | [source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Diabetic Retinopathy | image | Stage 2 | [diabetic_retinopathy.py](medical/adapters/diabetic_retinopathy.py) | [source](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized) |
| EBHI-Seg | image | Stage 2 | [ebhi_seg.py](medical/adapters/ebhi_seg.py) | [source](https://github.com/dataset-ninja/ebhi-seg) |
| Liver Ultrasound | image | Stage 2 | [liver_ultrasound.py](medical/adapters/liver_ultrasound.py) | [source](https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset) |
| MultiCare Images | image | Stage 2 | [multicare.py](medical/adapters/multicare.py) | [source](https://huggingface.co/datasets/openmed-community/multicare-images/tree/5c954c4fbf9abdcb55053488dab6c1ef142796b5) |
| MultiCare Case Images | image | Stage 2 | [multicare.py](medical/adapters/multicare.py) | [source](https://huggingface.co/datasets/openmed-community/multicare-case-images/tree/c8517124928d2fe3651ee6cb6c560fce66e02344) |
| NIH Chest X-ray | image | Stage 2 | [nih_chest_xray.py](medical/adapters/nih_chest_xray.py) | [source](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/tree/36778e3b0e4f4b4fad31d1728d6190f3eda5b543) |
| NCT-CRC-HE-100K | image | Stage 2 | [nct_crc_he.py](medical/adapters/nct_crc_he.py) | [source](https://zenodo.org/records/1214456) |
| MedPix-2 | image-text | Stage 2 | [medpix.py](medical/adapters/medpix.py) | [source](https://zenodo.org/records/12624810) |

## Vision - Cooldown & LCP

### Image-to-text

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| ChartNet | image-text | Cooldown & LCP | [chartnet/](datasets/chartnet/) | [source](https://huggingface.co/datasets/ibm-granite/ChartNet/tree/52832dd476c9a7a7b01c20245e952958cef1e2b2) |
| HQ-50K | image-text | Cooldown & LCP | [hq50k/](datasets/hq50k/) | [source](https://huggingface.co/datasets/YangQiee/HQ-50K/tree/1c501522f77594726b00b5943f1804c600e0230d) |
| NASA Images | image-text | Cooldown & LCP | [nasa/](datasets/nasa/) | [source](https://www.nasa.gov/) (scraped) |
| Smithsonian Open Access | image-text | Cooldown & LCP | [smithsonian/](datasets/smithsonian/) | [source](https://www.si.edu/openaccess) |
| Visual Genome | image-text | Cooldown | [visual_genome/](datasets/visual_genome/) | [source](https://huggingface.co/datasets/ranjaykrishna/visual_genome/tree/65bc9e7e7353fff750326c9523e384701934e530) |
| WAFFLE | image-text | Cooldown & LCP | [waffle/](datasets/waffle/) | [source](https://tau-vailab.github.io/WAFFLE/) |
| Swisstopo Maps | image-text | Cooldown & LCP | [swisstopo/](datasets/swisstopo/) | [source](https://www.geo.admin.ch/en/wms-available-services-an-data) |
| MINT-1T ArXiv (Cooldown) | image-text | Cooldown & LCP | [mint_arxiv/](datasets/mint_arxiv/) | [source](https://huggingface.co/datasets/mlfoundations/MINT-1T-ArXiv/tree/7c5b00ffd5b563071010c3bf2082b4a8f836eb72) |
| Crello | image-text | Cooldown | [crello/](datasets/crello/) | [source](https://huggingface.co/datasets/cyberagent/crello/tree/7997e2f434ee4aa73cf4cdf22c5954cb175872e1) |
| DOCCI | image-text | Cooldown | TODO | [source](https://huggingface.co/datasets/google/docci/tree/a0a43eaf34676ffd008fb6565dd8c2ba00d09100) |
| PixMo Point Explanations | image-text | Cooldown | [pixmo/](datasets/pixmo/) | [source](https://huggingface.co/datasets/allenai/pixmo-point-explanations/tree/08a566fa00747e4c1c7e8481c350763b469c209c) |

### Interleave

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| OWID Grapher Charts | image-text | Cooldown & LCP | [owid/](datasets/owid/) | [source](https://ourworldindata.org/) (scraped) |
| OWID Articles | image-text | Cooldown & LCP | [owid/](datasets/owid/) | [source](https://ourworldindata.org/) (scraped) |
| OWID Data Insights | image-text | Cooldown & LCP | [owid/](datasets/owid/) | [source](https://ourworldindata.org/) (scraped) |
| Argimi Finance 10K | document | Cooldown | [argimi_finance_10k/](datasets/argimi_finance_10k/) | [source](https://huggingface.co/datasets/artefactory/Argimi-Ardian-Finance-10k-text-image/tree/d019db455ff58bc14cae72422c4fc3ef0c301ea7) |
| DailyMed SPL | multimodal | Cooldown | [dailymed/](datasets/dailymed/), [dailymed_spl/](datasets/dailymed_spl/) | [source](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm) |

## Vision - SFT

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| Innovator-VL-Instruct-46M | image-text | SFT | [innovator/](datasets/innovator/), [innovator_vl/](datasets/innovator_vl/), [download_innovator_vl_46m.slurm](datasets/sft/download_innovator_vl_46m.slurm) | [source](https://huggingface.co/datasets/InnovatorLab/Innovator-VL-Instruct-46M) |
| SenseNova-SI-8M | image-text | SFT | [sensenova_si_8m/](datasets/sensenova_si_8m/) | [source](https://huggingface.co/datasets/sensenova/SenseNova-SI-8M) |
| Nemotron-Image-Training-v3 | image-text | SFT | [nemotron/](datasets/nemotron/), [nemotron_image_training_v3/](datasets/nemotron_image_training_v3/) | [source](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3) |
| MapTrace (SFT) | image-text | SFT | [maptrace/](datasets/maptrace/) | [source](https://huggingface.co/datasets/google/MapTrace/tree/8dd60adfde2f189768f27204c78ec44af07a67bf) |
| BigEarthNet (SFT) | image-text | SFT | [download_bigearthnet.slurm](datasets/sft/download_bigearthnet.slurm) | [source](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) |
| Google RSRCC | image-text | SFT | [google_rsrcc/](datasets/google_rsrcc/) | [source](https://huggingface.co/datasets/google/RSRCC) |
| MolmoPoint-GUISyn | image-text | SFT | [molmopoint_guisyn/](datasets/molmopoint_guisyn/) | [source](https://huggingface.co/datasets/allenai/MolmoPoint-GUISyn) |
| VDR Cooking Recipes | image-text | SFT | [download_vdr_cooking.slurm](datasets/sft/download_vdr_cooking.slurm) | [source](https://huggingface.co/datasets/racineai/VDR_Cooking_Recipes) |
| TCM Shizhen (SFT) | image-text | SFT | [download_tcm_shizhen.slurm](datasets/sft/download_tcm_shizhen.slurm) | [source](https://huggingface.co/datasets/FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT) |
| PixMo Ask Model Anything | image-text | SFT | [pixmo/](datasets/pixmo/), [preprocessing/](preprocessing/) | [source](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) |
| Path-VQA | image-text | SFT | [download_pathvqa.slurm](datasets/sft/download_pathvqa.slurm) | [source](https://huggingface.co/datasets/flaviagiammarino/path-vqa) |
| Molmo2 MultiImageQA (SFT) | image-text | SFT | [download_molmo2_multiimage.slurm](datasets/sft/download_molmo2_multiimage.slurm) | [source](https://huggingface.co/datasets/allenai/Molmo2-MultiImageQA/tree/f47ca3644d394b548be07a68d5a6fc0275924f08) |
| Molmo2 SynMultiImageQA (SFT) | image-text | SFT | [molmo2/](datasets/molmo2/), [molmo2_synmultiimageqa/](datasets/molmo2_synmultiimageqa/) | [source](https://huggingface.co/datasets/allenai/Molmo2-SynMultiImageQA) |
| GMAI-VL (permissive) | image-text | SFT | [gmai/](datasets/gmai/) | [source](https://huggingface.co/datasets/General-Medical-AI/GMAI-VL-5.5M) |
| PubMedVision | image-text | SFT | [pubmedvision/](datasets/pubmedvision/) | [source](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) |
| LLaVA-OneVision2 Spatial (OSD + RoboRef Sim) | image-text | SFT | [llava_onevision2_spatial/](datasets/llava_onevision2_spatial/) | [source](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-2-Data/tree/e73747a5) |
| FineVision - lnqa_recap | image-text | SFT | [redistill_lnqa.py](datasets/finevision/redistill_lnqa.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |
| FineVision - CoSyn_400k | image-text | SFT | [finevision/](datasets/finevision/) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |
| FineVision - mmevol (judged) | image-text | SFT | [redistill_mmevol_hybrid.py](datasets/finevision/redistill_mmevol_hybrid.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |
| FineVision - multihiertt | document | SFT | [clean_multihiertt.py](datasets/finevision/clean_multihiertt.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |
| FineVision - memotion (recap_en) | image-text | SFT | [redistill_memotion_qwen.py](datasets/finevision/redistill_memotion_qwen.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |
| FineVision - chinesememe (recap_en) | image-text | SFT | [redistill_chinesememe_qwen.py](datasets/finevision/redistill_chinesememe_qwen.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |
| FineVision - spatialsense (gold) | image-text | SFT | [redistill_spatialsense_gold.py](datasets/finevision/redistill_spatialsense_gold.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |
| LLaVA-OV Permissive | image-text | SFT | TODO | [source](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data) |
| PixMo Cap QA | image-text | SFT | [pixmo/](datasets/pixmo/) | [source](https://huggingface.co/datasets/allenai/pixmo-cap) |
| PixMo Point Explanations (SFT) | image-text | SFT | [pixmo/](datasets/pixmo/) | [source](https://huggingface.co/datasets/allenai/pixmo-point-explanations) |

## Audio - Stage 1

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| AudioSet (balanced + unbalanced) | audio | Stage 1 | [audio_set/](datasets/audio_set/) | [source](https://huggingface.co/datasets/agkphysics/AudioSet) |
| MTG-Jamendo | audio | Stage 1 | [mtg_jamendo/](datasets/mtg_jamendo/) | [source](https://huggingface.co/datasets/rkstgr/mtg-jamendo) |
| Suno | audio | Stage 1 | [suno/](datasets/suno/) | [source](https://huggingface.co/datasets/nyuuzyou/suno) |
| CommonVoice 24 | audio | Stage 1 | [commonvoice/](datasets/commonvoice/) | [source](https://commonvoice.mozilla.org/) |
| Unsupervised People's Speech | audio | Stage 1 | [peoples_speech/](datasets/peoples_speech/) | [source](https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech) |
| VoxPopuli | audio | Stage 1 | [voxpopuli/](datasets/voxpopuli/) | [source](https://github.com/facebookresearch/voxpopuli) |
| Gemeinderat Zurich | audio | Stage 1 | TODO | internal |

## Audio - Stage 2 / Cooldown

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| AISHELL-1/3/4 | audio | Stage 2 | [aishell/](datasets/aishell/), [aishell4/](datasets/aishell4/) | [source](https://www.openslr.org/33/) |
| Zeroth Korean | audio | Stage 2 | [zeroth_korean/](datasets/zeroth_korean/) | [source](https://huggingface.co/datasets/Bingsu/zeroth-korean) |
| People's Speech (supervised) | audio | Stage 2 | [peoples_speech/](datasets/peoples_speech/) | [source](https://huggingface.co/datasets/MLCommons/peoples_speech) |
| SPC-R | audio | Stage 2 | [spc_r_segmented/](datasets/spc_r_segmented/) | [source](https://huggingface.co/datasets/i4ds/spc_r) |
| WenetSpeech | audio | Stage 2 | [wenetspeech/](datasets/wenetspeech/) | [source](https://huggingface.co/datasets/wenet-e2e/wenetspeech) |
| CommonVoice (48-lang, Stage 2) | audio | Stage 2 | [commonvoice/](datasets/commonvoice/) | [source](https://commonvoice.mozilla.org/) |
| GigaSpeech | audio | Stage 2 | [gigaspeech/](datasets/gigaspeech/) | [source](https://huggingface.co/datasets/speechcolab/gigaspeech) |
| GigaSpeech 2 | audio | Stage 2 | [gigaspeech2/](datasets/gigaspeech2/) | [source](https://huggingface.co/datasets/speechcolab/gigaspeech2) |
| OmniLingual ASR | audio | Stage 2 | [omnilingual_asr/](datasets/omnilingual_asr/) | [source](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) |
| EuroSpeech | audio | Stage 2 | [eurospeech/](datasets/eurospeech/) | [source](https://huggingface.co/datasets/disco-eth/EuroSpeech) |
| Granary (YODAS, YTC, VoxPopuli, others) | audio | Stage 2 | [granary/](datasets/granary/) | [source](https://huggingface.co/datasets/espnet/yodas-granary) |
| ParlaSpeech-RS | audio | Stage 2 | [parlaspeech_rs/](datasets/parlaspeech_rs/) | [source](https://huggingface.co/datasets/classla/ParlaSpeech-RS) |
| ParlaMent Parla | audio | Stage 2 | [parlament_parla/](datasets/parlament_parla/) | [source](https://huggingface.co/datasets/projecte-aina/parlament_parla_v3) |
| SeamlessAlign | audio | Stage 2 | [seamless_align/](datasets/seamless_align/) | [source](https://huggingface.co/datasets/ai4bharat/SeamlessAlign) |
| Kathbath | audio | Stage 2 | [kathbath/](datasets/kathbath/) | [source](https://huggingface.co/datasets/ai4bharat/Kathbath) |
| Coral V3 | audio | Stage 2 | [coral/](datasets/coral/) | [source](https://huggingface.co/datasets/CoRal-project/coral-v3) |
| LegCo Speech | audio | Stage 2 | [legco_speech/](datasets/legco_speech/) | [source](https://huggingface.co/datasets/laubonghaudoi/legco-speech) |
| MultiMed | audio | Stage 2 | [multimed/](datasets/multimed/) | [source](https://huggingface.co/datasets/leduckhai/MultiMed) |
| LibriHeavy | audio | Stage 2 | [libriheavy/](datasets/libriheavy/) | [source](https://huggingface.co/datasets/mythicinfinity/libriheavy) |
| VietSpeech | audio | Stage 2 | [vietspeech/](datasets/vietspeech/) | [source](https://huggingface.co/datasets/NhutP/VietSpeech) |
| Infore2 Audiobooks | audio | Stage 2 | [infore2_audiobooks/](datasets/infore2_audiobooks/) | [source](https://huggingface.co/datasets/doof-ferb/infore2_audiobooks) |
| ViMedCSS | audio | Stage 2 | [vimedcss/](datasets/vimedcss/) | [source](https://huggingface.co/datasets/tensorxt/ViMedCSS) |
| Kazakh Speech | audio | Stage 2 | [kazakh/](datasets/kazakh/), [kazakh_speech/](datasets/kazakh_speech/) | [source](https://huggingface.co/datasets/Flamme-VRM/kazakh-speech-dataset) |
| MLS | audio | Stage 2 | [mls/](datasets/mls/) | [source](https://huggingface.co/datasets/facebook/multilingual_librispeech) |
| F1 Team Radio | audio | Stage 2 | [f1_team_radio/](datasets/f1_team_radio/) | [source](https://huggingface.co/datasets/MikCil/f1-team-radio) |
| Zoengjyutgaai | audio | Stage 2 | [zoengjyutgaai/](datasets/zoengjyutgaai/) | [source](https://huggingface.co/datasets/CanCLID/zoengjyutgaai) |
| Emilia/YODAS | audio | Stage 2 | [Emilia_YODAS/](datasets/Emilia_YODAS/) | [source](https://huggingface.co/datasets/amphion/Emilia-Dataset) |
| HUI-Audio-Corpus-German | audio | Stage 2 | [hui_audio_corpus_german/](datasets/hui_audio_corpus_german/), [hui_audio_german/](datasets/hui_audio_german/) | [source](https://opendata.iisys.de/opendata/Datasets/HUI-Audio-Corpus-German) |
| Russian LibriSpeech | audio | Cooldown | [russian_librispeech/](datasets/russian_librispeech/) | [source](https://www.openslr.org/96/) |
| SloPalSpeech | audio | Cooldown | [slopalspeech/](datasets/slopalspeech/) | [source](https://huggingface.co/datasets/NaiveNeuron/SloPalSpeech) |
| ToneWebinars | audio | Cooldown | [tonewebinars/](datasets/tonewebinars/) | [source](https://huggingface.co/datasets/Vikhrmodels/ToneWebinars) |
| Tuda-De | audio | Cooldown | [tuda_de/](datasets/tuda_de/) | [source](https://huggingface.co/datasets/uhhlt/Tuda-De) |
| MediaSpeech | audio | Cooldown | [mediaspeech/](datasets/mediaspeech/), [openslr108_mediaspeech/](datasets/openslr108_mediaspeech/) | [source](https://www.openslr.org/108/) |
| Farsi ASR | audio | Cooldown | [farsi_asr/](datasets/farsi_asr/) | [source](https://huggingface.co/datasets/farsi-asr/farsi-asr-dataset) |
| Aozora Hurigana | audio | Cooldown | [aozora_hurigana/](datasets/aozora_hurigana/) | [source](https://github.com/ndl-lab/hurigana-speech-corpus-aozora) |
| Ghana English Speech | audio | Cooldown | [ghana_english/](datasets/ghana_english/) | [source](https://huggingface.co/datasets/ghananlpcommunity/ghana-english-speech-600hrs) |
| CC-Podcasts | audio | Cooldown | [ccpodcasts/](datasets/ccpodcasts/) | [source](https://huggingface.co/datasets/shuyuej/CC-Podcasts) |
| Localized Narratives (audio) | audio | Cooldown | [localized_narratives/](datasets/localized_narratives/) | [source](https://storage.googleapis.com/localized-narratives) |
| MRSAudio | audio | Stage 2 | [mrsaudio/](datasets/mrsaudio/) | [source](https://huggingface.co/datasets/MRSAudio/MRSAudio) |

## Audio - SFT

| Dataset | Modality | Stage | Processing | Upstream |
|---|---|---|---|---|
| Marco Longspeech | audio | SFT | [marco_longspeech/](datasets/marco_longspeech/) | [source](https://huggingface.co/datasets/AIDC-AI/Marco_Longspeech) |
| AudioMCQ StrongAC GeminiCoT | audio | SFT | [audiomcq_strongac_cot/](datasets/audiomcq_strongac_cot/) | [source](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT) |
| TeleAntiFraud | audio | SFT | [teleantifraud/](datasets/teleantifraud/) | [source](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud) |
| VoiceAssistant-400K | audio | SFT | [voiceassistant_400k/](datasets/voiceassistant_400k/) | [source](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) |
| HeySQuAD (human + machine) | audio | SFT | [heysquad/](datasets/heysquad/) | [source](https://huggingface.co/datasets/yijingwu/HeySQuAD_human) |
| TCM Shizhen Speech (SFT) | audio | SFT | TODO | [source](https://huggingface.co/datasets/FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT) |
| Vocalized SFT | audio | SFT | TODO | internal |
