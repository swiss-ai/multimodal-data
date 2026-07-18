# Datasets

Source of truth: [`summary.yaml`](summary.yaml). This table is generated - edit the YAML, then run `python datasets/generate_readme.py`.

`Comment` documents license/subset filtering (which Mixed/NC/SA parts were removed or which permissive subset was kept): a linked path points to the filtering code, `TODO` marks filtering still to be documented.

| Dataset | License | Modality | Stage | Processing | Upstream | Comment |
|---|---|---|---|---|---|---|
| TreeOfLife-10M | Mixed | image | Stage 1 | [tree_of_life](tree_of_life/) | [source](https://huggingface.co/datasets/imageomics/TreeOfLife-10M/tree/91debffb7146c32c89d76feb1eb575b555e2ecc7) | [main.py](tree_of_life/main.py) - mapped each image to its per-image license (from licenses.csv) and kept only CC0/CC-BY/CC-BY-3.0/CC-BY-4.0/public-domain |
| LAION Aesthetics 12M UMAP | MIT | image | Stage 1 | [laion](laion/), [laion.py](../medical/adapters/laion.py) | [source](https://huggingface.co/datasets/dclure/laion-aesthetics-12m-umap/tree/06928317703bcfa6099c7fc0f13e11bb295e7769) |  |
| MINT-1T HTML | CC-BY-4.0 | image-text | Stage 1 | [mint_html](mint_html/) | [source](https://huggingface.co/datasets/mlfoundations/MINT-1T-HTML/tree/906a8b85cea61198ff7339c4dd711ad0b5361847) |  |
| MINT-1T PDF | CC-BY-4.0 | image | Stage 1 | [mint_arxiv](mint_arxiv/) | [source](https://huggingface.co/collections/mlfoundations/mint-1t) |  |
| MINT-1T ArXiv | CC-BY-4.0 | image | Stage 1 | [mint_arxiv](mint_arxiv/) | [source](https://huggingface.co/datasets/mlfoundations/MINT-1T-ArXiv/tree/7c5b00ffd5b563071010c3bf2082b4a8f836eb72) |  |
| BigDocs-7.5M | CC-BY-4.0 | image | Stage 1 | [bigdocs](bigdocs/) | [source](https://huggingface.co/datasets/ServiceNow/BigDocs-7.5M/tree/dae4403c28307bd5328920740e81ce5232819e74) |  |
| SWISSIMAGE 10cm | OGD | image | Stage 1, Cooldown & LCP | [swissimage](swissimage/), [swisstopo](swisstopo/) | [source](https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10#Download) |  |
| Copernicus-Bench (bigearthnet, dfc2020, eurosat) | Mixed | image | Stage 1 | [copernicus](copernicus/) | [source](https://huggingface.co/datasets/wangyi111/Copernicus-Bench/tree/a287ab1b414d2bff99557166988571c5885ed81a) | [main.py](copernicus/main.py) - ingested only the three non-NC/SA subsets: BigEarthNet (CDLA-Permissive), EuroSAT (MIT), DFC2020 |
| MMammoTH | Apache-2.0 | image-text | Stage 1 | [preprocessing](../preprocessing/) | [source](https://huggingface.co/datasets/MAmmoTH-VL/MAmmoTH-VL-Instruct-12M/tree/bac8f77cb8a8f9c4d0de407c6e3a589bd722562a) |  |
| HoloAssist | CDLA-Permissive-2 | image | Stage 1 | [holoassist.py](../medical/adapters/holoassist.py), [apertus1p5](../medical/configs/apertus1p5/) | [source](https://holoassist.github.io/) |  |
| Nicola Handwriting - Docs | Apache 2.0 (in house generated) | image | Stage 1 | [handwritting_data_processing_scripts](https://github.com/swiss-ai/handwritting_data_processing_scripts) | [source](https://huggingface.co/datasets/handwriting-apertus/handwriting_data) (internal) |  |
| Nicola Handwriting - Slides | Apache 2.0 (in house generated) | image | Stage 1 | [handwritting_data_processing_scripts](https://github.com/swiss-ai/handwritting_data_processing_scripts) | [source](https://huggingface.co/datasets/handwriting-apertus/handwriting_data) (internal) |  |
| Breast Ultrasound (BUSI) | Unknown | image | Stage 1, Stage 2 | [preprocessing](../preprocessing/) | [source](https://www.kaggle.com/datasets/sabahesaraki/breast-ultrasound-images-dataset) |  |
| COVID-US | Mixed, Unknown | image | Stage 1, Stage 2 | [preprocessing](../preprocessing/) | [source](https://github.com/nrc-cnrc/COVID-US) | dropped non-NC/SA data sources |
| DDTI Thyroid Ultrasound | Unknown | image | Stage 1, Stage 2 | [preprocessing](../preprocessing/) | [source](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) |  |
| ISIC Skin Disease | Apache-2.0 | image | Stage 1 | [isic.py](../medical/adapters/isic.py) | [source](https://www.kaggle.com/datasets/abhii1929/isic-skin-disease-image-dataset-4-classes) |  |
| MedMax | Mixed | image-text | Stage 1, Stage 2 | [medmax.py](../medical/adapters/medmax.py), [preprocessing](../preprocessing/) | [source](https://huggingface.co/datasets/mint-medmax/medmax_data/tree/2a63fb4d5e57bbf30a130976825fadf95ea95ebb) | [medmax.py](../medical/adapters/medmax.py) - each image mapped by its `source` dataset via the metadata lookup, non-permissive samples were dropped |
| MedMNIST | CC-BY-4.0 | image | Stage 1 | [medmnist.py](../medical/adapters/medmnist.py) | [source](https://zenodo.org/records/10519652) |  |
| MedTrinity-25M | Mixed | image-text | Stage 1, Stage 2 | [medtrinity.py](../medical/adapters/medtrinity.py), [preprocessing](../preprocessing/) | [source](https://huggingface.co/datasets/UCSC-VLAA/MedTrinity-25M/tree/89e5c684794e5c4cc1af9e8f1a7798af7c937dbf) | [medtrinity.py](../medical/adapters/medtrinity.py) - kept only rows whose `source` is in `allowed_sources` list (deeplesion, brats, pmc_oa, NCT-CRC-HE-100K, nih_chest, ...) |
| OpenPMC-18M | MIT | image | Stage 1, Stage 2 | [open_pmc_18m.py](../medical/adapters/open_pmc_18m.py) | [source](https://huggingface.co/datasets/vector-institute/open-pmc-18m/tree/b5a67783ec3e1bf91809a5efc4b72fbedacacdf6) |  |
| PMC OA | Mixed | image | Stage 1, Stage 2 | [pmc_oa.py](../medical/adapters/pmc_oa.py), [preprocessing](../preprocessing/) | [source](https://huggingface.co/datasets/axiong/pmc_oa/tree/1d2296e9c022a24e82a47e524d53f0915b98c926) | sourced from the PMC Open Access commercial-use subset (CC-BY/CC0 articles only), non-commercial OA is excluded upstream |
| RFMiD 2 | CC-BY-4.0 | image | Stage 1 | [rfmid2.py](../medical/adapters/rfmid2.py) | [source](https://zenodo.org/records/7505822) |  |
| SciN | Other | image | Stage 1 | [scin.py](../medical/adapters/scin.py) | [source](https://huggingface.co/datasets/google/scin/tree/996257142f7517fb8991a28cfba46ec4e3f530a9) | bespoke SCIN Data Use License (modified CC BY) |
| SLID-E | MIT, CC-BY-4.0 | image | Stage 1 | [slide.py](../medical/adapters/slide.py) | [source](https://figshare.com/articles/dataset/SLID-E/26172919) |  |
| Open UFI | CC-BY | image | Stage 1 | [uwf.py](../medical/adapters/uwf.py) | [source](https://springernature.figshare.com/articles/dataset/Open_UFI_and_clinical_IQA/26936446) |  |
| Recap-DataComp-1B | CC-BY-4.0 | image-text | Stage 2, Cooldown | [recap_datacomp](recap_datacomp/), [recap_datacomp_1b](recap_datacomp_1b/) | [source](https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B/tree/457c44d98651bcfdfb3cc8695f5e60a0d2705e78) |  |
| LaTeX Formulas 80M | Apache-2.0 | image-text | Stage 2, Cooldown | [latex_formulas_80m](latex_formulas_80m/) | [source](https://huggingface.co/datasets/OleehyO/latex-formulas-80M/tree/5cd783320b0092caa85720a85d86595f42df043b) |  |
| BLIP3 Grounding 50M | Apache-2.0 | image-text | Stage 2 | [blip3_grounding](blip3_grounding/) | [source](https://huggingface.co/datasets/Salesforce/blip3-grounding-50m/tree/4d622c4f19b8a6b91ad914caf343306e363ff79b) |  |
| PD12M Full | CDLA-Permissive-2.0 | image-text | Stage 2 | [pd12m](pd12m/) | [source](https://huggingface.co/datasets/Spawning/pd12m-full/tree/7bda6bfe13c6a39fd0a286adced5df1228041720) |  |
| FaceCaption-15M | CC-BY-4.0 | image-text | Stage 2 | [facecaption_15m](facecaption_15m/), [facetaption](facetaption/) | [source](https://huggingface.co/datasets/OpenFace-CQUPT/FaceCaption-15M/tree/3ed92d90f7fc7199b47c4da17c6863b1a175f380) |  |
| Megalith-10M | MIT | image-text | Stage 2 | [megalith](megalith/) | [source](https://huggingface.co/datasets/madebyollin/megalith-10m/tree/1e65a79953396f6d05f60eba4cc564541ad4be8c) |  |
| SkyScript | MIT | image-text | Stage 2, Cooldown | [skyscript](skyscript/) | [source](https://github.com/wangzhecheng/SkyScript) |  |
| WebSight V0.2 | CC-BY-4.0 | image-text | Stage 2 | [websight](websight/) | [source](https://huggingface.co/datasets/HuggingFaceM4/WebSight/tree/b11f8172f89c992b56ac702319e02c428cca4a4e) |  |
| Open Images V7 | CC-BY | image-text | Stage 2 | [openimages](openimages/) | [source](https://huggingface.co/datasets/bitmind/open-images-v7/tree/4518ecd40f8f9ef66ee4356be438f840c714e95a) |  |
| DaTikZ-V4 | Apache-2.0 | image-text | Stage 2 | [datikz_v4](datikz_v4/) | [source](https://huggingface.co/datasets/nllg/DaTikZ-V4/tree/33734c83608211682be11001a1618856fc1979dd) |  |
| Art Museums PD 440k | CC-BY-4.0 | image-text | Stage 2, Cooldown | [art_museums_pd](art_museums_pd/) | [source](https://huggingface.co/datasets/Mitsua/art-museums-pd-440k/tree/fba945da78b36262eb9272067197cc28d06cffbf) |  |
| Fine-T2I | Apache-2.0 | image-text | Stage 2 | [fine_t2i](fine_t2i/) | [source](https://huggingface.co/datasets/ma-xu/fine-t2i/tree/28fdd5663ee202b5cafc01d6ed08a03f14957854) |  |
| MapTrace | CC-BY-4.0 | image-text | Stage 2 | [maptrace](maptrace/) | [source](https://huggingface.co/datasets/google/MapTrace/tree/8dd60adfde2f189768f27204c78ec44af07a67bf) |  |
| FLAIR-HUB | Etalab Open License 2.0 | image | Stage 2 | [flair_hub](flair_hub/), [ign](ign/) | [source](https://huggingface.co/datasets/IGNF/FLAIR-HUB/tree/8275163f72f0eed69050a703925791b7c3577f10) |  |
| IGN City Tiles | Etalab Open License 2.0 | image | Stage 2 | [ign](ign/) | [source](https://geoservices.ign.fr/planign) |  |
| Shopify Product Catalogue | Apache-2.0 | image-text | Stage 2 | [product_catalogue](product_catalogue/) | [source](https://huggingface.co/datasets/Shopify/product-catalogue/tree/d5c517c509f5aca99053897ef1de797d6d7e5aa5) |  |
| EgoPAT3Dv2 | Unknown | image | Stage 2, Cooldown | [egopat3dv2](egopat3dv2/) | [source](https://huggingface.co/datasets/ai4ce/EgoPAT3Dv2/tree/9f20d0b0f6f48022bc2e10c46f219e3b89c44681) |  |
| LLaVA-OV Mid-Training 85M | Apache-2.0 | image-text | Stage 2 | - | [source](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/tree/c5218cad785eba7d218137e8ce4997bda568a050) |  |
| BLIP3o Long Caption | Apache-2.0 | image-text | Stage 2 | - | [source](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption/tree/e4d07091a466d1a1e35a9b0c61caddc78d14a059) |  |
| CommonCatalog CC-BY | Mixed | image-text | Stage 2 | [loader_commoncatalog.py](../recaption/vllm/loader_commoncatalog.py) | [source](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by/tree/80f50fe4a1ca937f37a11be3f8eee5199d776ff3) | [loader_commoncatalog.py](../recaption/vllm/loader_commoncatalog.py) - loader_commoncatalog.py keeps only rows whose `licenseurl` is CC-BY (ALLOWED_LICENSE_URLS) |
| TextAtlas5M | Mixed | image-text | Stage 2 | - | [source](https://huggingface.co/datasets/CSU-JPG/TextAtlas5M/tree/f9f2a0f5000fbb078f718197acb45cfb9ceed551) | subsets carry mixed licenses, kept only the non-NC/SA (MIT-licensed) subsets, dropped the rest |
| UNO-1M | Apache-2.0 | image-text | Stage 2 | [loader_uno_1m_v3.py](../recaption/vllm/loader_uno_1m_v3.py) | [source](https://huggingface.co/datasets/bytedance-research/UNO-1M/tree/f25bb61db6d6d66d82f41d1e613c0e04ba342e84) |  |
| RSTeller | Apache-2.0 | image-text | Stage 2 | [rsteller.py](../image_text/rsteller.py) | [source](https://huggingface.co/datasets/SlytherinGe/RSTeller/tree/a03b35f1bc9a3ac14ae93724d175c2611f1bba5b) |  |
| GeoChat Instruct | Apache-2.0 | image-text | Stage 2 | [geochat.py](../image_text/geochat.py) | [source](https://huggingface.co/datasets/MBZUAI/GeoChat_Instruct/tree/8eb13307eabc7fa9c1f8b0e61e372a327ccd68b1) |  |
| MIT-10M | Apache-2.0 | image-text | Stage 2 | [loader_mit_10m_qwen_from_text.py](../recaption/vllm/loader_mit_10m_qwen_from_text.py) | [source](https://huggingface.co/datasets/liboaccn/MIT-10M/tree/bcba6b2651771c69f93e000486c2baa0896d32c3) |  |
| PixMo-Cap | ODC-BY-1.0 | image-text | Stage 2, Cooldown | [pixmo](pixmo/), [pixmo_cap](../download/special/pixmo_cap/), [preprocessing](../preprocessing/) | [source](https://huggingface.co/datasets/allenai/pixmo-cap/tree/edce6390d9d5be6c8db0d863fbe62718c88988a4) |  |
| PIN-200M | Apache-2.0 | image-text | Stage 2, Cooldown & LCP | [pin_200m](pin_200m/) | [source](https://huggingface.co/datasets/m-a-p/PIN-200M/tree/f69c5da58f4284c6687a0e058c21e67fca9a1b66) |  |
| TCM Shizhen (Book Vision + Web Vision) | Apache-2.0 | image-text | Stage 2, Cooldown & LCP | [tcm_pretrain_shizhen](tcm_pretrain_shizhen/) | [source](https://huggingface.co/datasets/FreedomIntelligence/TCM-Pretrain-Data-ShizhenGPT/tree/db4874ce4e322f47432fe322c558e516c5aad71e) |  |
| Molmo2 Syn MultiImage | ODC-By | image-text | Stage 2, Cooldown | [molmo2](molmo2/), [molmo2_synmultiimageqa](molmo2_synmultiimageqa/) | [source](https://huggingface.co/datasets/allenai/Molmo2-MultiImageQA/tree/f47ca3644d394b548be07a68d5a6fc0275924f08) |  |
| Brain Tumor MRI | CC-BY-4.0 | image | Stage 2 | [brain_tumor_mri.py](../medical/adapters/brain_tumor_mri.py) | [source](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |  |
| Diabetic Retinopathy | CC0 1.0 Universal | image | Stage 2 | [diabetic_retinopathy.py](../medical/adapters/diabetic_retinopathy.py) | [source](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-2015-data-colored-resized) |  |
| EBHI-Seg | CC-BY-4.0 | image | Stage 2 | [ebhi_seg.py](../medical/adapters/ebhi_seg.py) | [source](https://github.com/dataset-ninja/ebhi-seg) |  |
| Liver Ultrasound | CC-BY-4.0 | image | Stage 2 | [liver_ultrasound.py](../medical/adapters/liver_ultrasound.py) | [source](https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset) |  |
| MultiCare Images | CC-BY-4.0 | image | Stage 2 | [multicare.py](../medical/adapters/multicare.py) | [source](https://huggingface.co/datasets/openmed-community/multicare-images/tree/5c954c4fbf9abdcb55053488dab6c1ef142796b5) |  |
| MultiCare Case Images | Mixed | image | Stage 2 | [multicare.py](../medical/adapters/multicare.py) | [source](https://huggingface.co/datasets/openmed-community/multicare-case-images/tree/c8517124928d2fe3651ee6cb6c560fce66e02344) | images carry per-article PMC CC licenses, samples taken from (pre-filtered) permissive articles |
| NIH Chest X-ray | CC0 1.0 Universal | image | Stage 2 | [nih_chest_xray.py](../medical/adapters/nih_chest_xray.py) | [source](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/tree/36778e3b0e4f4b4fad31d1728d6190f3eda5b543) |  |
| NCT-CRC-HE-100K | CC-BY-4.0 | image | Stage 2 | [nct_crc_he.py](../medical/adapters/nct_crc_he.py) | [source](https://zenodo.org/records/1214456) |  |
| MedPix-2 | CC-BY-4.0 | image-text | Stage 2 | [medpix.py](../medical/adapters/medpix.py) | [source](https://zenodo.org/records/12624810) |  |
| ChartNet | Mixed | image-text | Cooldown & LCP | [chartnet](chartnet/) | [source](https://huggingface.co/datasets/ibm-granite/ChartNet/tree/52832dd476c9a7a7b01c20245e952958cef1e2b2) | [download.slurm](chartnet/download.slurm) - download.slurm fetches only the `core_permissive/*` subset (CDLA-Permissive-2.0) |
| HQ-50K | CC-BY-4.0 | image-text | Cooldown & LCP | [hq50k](hq50k/) | [source](https://huggingface.co/datasets/YangQiee/HQ-50K/tree/1c501522f77594726b00b5943f1804c600e0230d) |  |
| NASA Images | Unknown | image-text | Cooldown & LCP | [nasa](nasa/) | [source](https://www.nasa.gov/) (scraped) | [filter_cooldown.py](nasa/filter_cooldown.py) - filter_cooldown.py hard-filters to `license == "PD"` (public domain), ESA-joint and unclear-license images are dropped |
| Smithsonian Open Access | CC0 | image-text | Cooldown & LCP | [smithsonian](smithsonian/) | [source](https://www.si.edu/openaccess) |  |
| Visual Genome | CC-BY-4.0 | image-text | Cooldown | [visual_genome](visual_genome/) | [source](https://huggingface.co/datasets/ranjaykrishna/visual_genome/tree/65bc9e7e7353fff750326c9523e384701934e530) |  |
| WAFFLE | Unknown | image-text | Cooldown & LCP | [waffle](waffle/) | [source](https://tau-vailab.github.io/WAFFLE/) | [convert_to_parquet.py](waffle/convert_to_parquet.py) - convert_to_parquet.py classifies each image license into permissive (PD/CC0/CC-BY) vs sa buckets and drops NC/ND/unknown, only the permissive bucket is used |
| Swisstopo Maps | Open Government Data | image-text | Cooldown & LCP | [swisstopo](swisstopo/) | [source](https://www.geo.admin.ch/en/wms-available-services-an-data) |  |
| MINT-1T ArXiv (Cooldown) | CC-BY-4.0 | image-text | Cooldown & LCP | [mint_arxiv](mint_arxiv/) | [source](https://huggingface.co/datasets/mlfoundations/MINT-1T-ArXiv/tree/7c5b00ffd5b563071010c3bf2082b4a8f836eb72) |  |
| Crello | CDLA-Permissive-2.0 | image-text | Cooldown | [crello](crello/) | [source](https://huggingface.co/datasets/cyberagent/crello/tree/7997e2f434ee4aa73cf4cdf22c5954cb175872e1) |  |
| DOCCI | CC-BY-4.0 | image-text | Cooldown | - | [source](https://huggingface.co/datasets/google/docci/tree/a0a43eaf34676ffd008fb6565dd8c2ba00d09100) |  |
| PixMo Point Explanations | ODC-By | image-text | Cooldown | [pixmo](pixmo/) | [source](https://huggingface.co/datasets/allenai/pixmo-point-explanations/tree/08a566fa00747e4c1c7e8481c350763b469c209c) |  |
| OWID Grapher Charts | Unknown | image-text | Cooldown & LCP | [owid](owid/) | [source](https://ourworldindata.org/) (scraped) | [download_graphers.py](owid/download_graphers.py) - OWID content is CC-BY-4.0 by site policy |
| OWID Articles | Unknown | image-text | Cooldown & LCP | [owid](owid/) | [source](https://ourworldindata.org/) (scraped) | OWID content is CC-BY-4.0 by site policy |
| OWID Data Insights | Unknown | image-text | Cooldown & LCP | [owid](owid/) | [source](https://ourworldindata.org/) (scraped) | OWID content is CC-BY-4.0 by site policy |
| Argimi Finance 10K | CC-BY-4.0 | image-text | Cooldown | [argimi_finance_10k](argimi_finance_10k/) | [source](https://huggingface.co/datasets/artefactory/Argimi-Ardian-Finance-10k-text-image/tree/d019db455ff58bc14cae72422c4fc3ef0c301ea7) |  |
| DailyMed SPL | Unknown | image-text | Cooldown | [dailymed](dailymed/), [dailymed_spl](dailymed_spl/) | [source](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm) | DailyMed SPLs are FDA / US-government public-domain records, the whole release is non-NC/SA, no license filtering needed. |
| Innovator-VL-Instruct-46M | MIT | image-text | SFT | [innovator](innovator/), [innovator_vl](innovator_vl/), [download_innovator_vl_46m.slurm](sft/download_innovator_vl_46m.slurm) | [source](https://huggingface.co/datasets/InnovatorLab/Innovator-VL-Instruct-46M) |  |
| SenseNova-SI-8M | Apache-2.0 | image-text | SFT | [sensenova_si_8m](sensenova_si_8m/) | [source](https://huggingface.co/datasets/sensenova/SenseNova-SI-8M) |  |
| Nemotron-Image-Training-v3 | CC-BY-4.0 | image-text | SFT | [nemotron](nemotron/), [nemotron_image_training_v3](nemotron_image_training_v3/) | [source](https://huggingface.co/datasets/nvidia/Nemotron-Image-Training-v3) |  |
| MapTrace (SFT) | CC-BY-4.0 | image-text | SFT | [maptrace](maptrace/) | [source](https://huggingface.co/datasets/google/MapTrace/tree/8dd60adfde2f189768f27204c78ec44af07a67bf) |  |
| BigEarthNet (SFT) | CDLA-Permissive-1.0 | image-text | SFT | [download_bigearthnet.slurm](sft/download_bigearthnet.slurm) | [source](https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt) |  |
| Google RSRCC | Apache-2.0 | image-text | SFT | [google_rsrcc](google_rsrcc/) | [source](https://huggingface.co/datasets/google/RSRCC) |  |
| MolmoPoint-GUISyn | Apache-2.0 | image-text | SFT | [molmopoint_guisyn](molmopoint_guisyn/) | [source](https://huggingface.co/datasets/allenai/MolmoPoint-GUISyn) |  |
| VDR Cooking Recipes | Apache-2.0 | image-text | SFT | [download_vdr_cooking.slurm](sft/download_vdr_cooking.slurm) | [source](https://huggingface.co/datasets/racineai/VDR_Cooking_Recipes) |  |
| TCM Shizhen (SFT) | Apache-2.0 | image-text | SFT | [download_tcm_shizhen.slurm](sft/download_tcm_shizhen.slurm) | [source](https://huggingface.co/datasets/FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT) |  |
| PixMo Ask Model Anything | ODC-By | image-text | SFT | [pixmo](pixmo/), [preprocessing](../preprocessing/) | [source](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything) |  |
| Path-VQA | MIT | image-text | SFT | [download_pathvqa.slurm](sft/download_pathvqa.slurm) | [source](https://huggingface.co/datasets/flaviagiammarino/path-vqa) |  |
| Molmo2 MultiImageQA (SFT) | ODC-By | image-text | SFT | [download_molmo2_multiimage.slurm](sft/download_molmo2_multiimage.slurm) | [source](https://huggingface.co/datasets/allenai/Molmo2-MultiImageQA/tree/f47ca3644d394b548be07a68d5a6fc0275924f08) |  |
| Molmo2 SynMultiImageQA (SFT) | ODC-By | image-text | SFT | [molmo2](molmo2/), [molmo2_synmultiimageqa](molmo2_synmultiimageqa/) | [source](https://huggingface.co/datasets/allenai/Molmo2-SynMultiImageQA) |  |
| GMAI-VL (permissive) | Mixed | image-text | SFT | [gmai](gmai/) | [source](https://huggingface.co/datasets/General-Medical-AI/GMAI-VL-5.5M) | [gmai](gmai/) - kept only the permissive source datasets, NC/SA constituents dropped |
| PubMedVision | Apache-2 | image-text | SFT | [pubmedvision](pubmedvision/) | [source](https://huggingface.co/datasets/FreedomIntelligence/PubMedVision) |  |
| LLaVA-OneVision2 Spatial (OSD + RoboRef Sim) | Apache-2 | image-text | SFT | [llava_onevision2_spatial](llava_onevision2_spatial/) | [source](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-2-Data/tree/e73747a5) |  |
| FineVision - lnqa_recap | CC-BY-4.0 | image-text | SFT | [redistill_lnqa.py](finevision/redistill_lnqa.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |  |
| FineVision - CoSyn_400k | CC-BY-4.0 | image-text | SFT | [finevision](finevision/) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |  |
| FineVision - mmevol (judged) | CC-BY-4.0 | image-text | SFT | [redistill_mmevol_hybrid.py](finevision/redistill_mmevol_hybrid.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |  |
| FineVision - multihiertt | CC-BY-4.0 | image-text | SFT | [clean_multihiertt.py](finevision/clean_multihiertt.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |  |
| FineVision - memotion (recap_en) | CC-BY-4.0 | image-text | SFT | [redistill_memotion_qwen.py](finevision/redistill_memotion_qwen.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |  |
| FineVision - chinesememe (recap_en) | CC-BY-4.0 | image-text | SFT | [redistill_chinesememe_qwen.py](finevision/redistill_chinesememe_qwen.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |  |
| FineVision - spatialsense (gold) | CC-BY-4.0 | image-text | SFT | [redistill_spatialsense_gold.py](finevision/redistill_spatialsense_gold.py) | [source](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |  |
| LLaVA-OV Permissive | Apache-2 | image-text | SFT | - | [source](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data) |  |
| PixMo Cap QA | ODC-BY-1.0 | image-text | SFT | [pixmo](pixmo/) | [source](https://huggingface.co/datasets/allenai/pixmo-cap) |  |
| PixMo Point Explanations (SFT) | ODC-By | image-text | SFT | [pixmo](pixmo/) | [source](https://huggingface.co/datasets/allenai/pixmo-point-explanations) |  |
| AudioSet (balanced + unbalanced) | CC BY 4.0 | audio | Stage 1 | [audio_set](audio_set/) | [source](https://huggingface.co/datasets/agkphysics/AudioSet) |  |
| MTG-Jamendo | Apache-2.0 | audio | Stage 1 | [mtg_jamendo](mtg_jamendo/) | [source](https://huggingface.co/datasets/rkstgr/mtg-jamendo) |  |
| Suno | CC-BY-1.0 | audio | Stage 1 | [suno](suno/) | [source](https://huggingface.co/datasets/nyuuzyou/suno) |  |
| CommonVoice 24 | CC0-1.0 | audio | Stage 1 | [commonvoice](commonvoice/) | [source](https://commonvoice.mozilla.org/) |  |
| Unsupervised People's Speech | CC-BY-SA,  CC-BY | audio | Stage 1 | [peoples_speech](peoples_speech/) | [source](https://huggingface.co/datasets/MLCommons/unsupervised_peoples_speech) | [license_filter.py](peoples_speech/unsupervised/license_filter.py), [generate_commercial_ids.py](peoples_speech/unsupervised/generate_commercial_ids.py) - generate_commercial_ids.py selects archive IDs with commercial-use licenses (PD/CC0/US-Gov/CC-BY, excluding SA/NC), license_filter.py rewrites the shards to keep only those samples |
| VoxPopuli | Mixed, CC0 | audio | Stage 1 | [voxpopuli](voxpopuli/) | [source](https://github.com/facebookresearch/voxpopuli) | uses only the CC0 raw-audio "VoxPopuli Data" tarballs, the other subsets under other terms are excluded |
| Gemeinderat Zurich | MIT | audio | Stage 1 | - | [source](https://www.cs.technik.fhnw.ch/i4ds-datasets) |  |
| AISHELL-1/3/4 | Apache-2.0 | audio | Stage 2 | [aishell](aishell/), [aishell4](aishell4/) | [source](https://www.openslr.org/33/) |  |
| Zeroth Korean | CC-BY-4.0 | audio | Stage 2 | [zeroth_korean](zeroth_korean/) | [source](https://huggingface.co/datasets/Bingsu/zeroth-korean) |  |
| People's Speech (supervised) | CC-BY-SA,  CC-BY | audio | Stage 2 | [peoples_speech](peoples_speech/) | [source](https://huggingface.co/datasets/MLCommons/peoples_speech) | uses only the CC-BY configs (clean + dirty), the CC-BY-SA subsets (clean_sa/dirty_sa) are excluded. |
| SPC-R | CC BY 4.0 | audio | Stage 2 | [spc_r_segmented](spc_r_segmented/) | [source](https://huggingface.co/datasets/i4ds/spc_r) |  |
| WenetSpeech | CC-BY-4 | audio | Stage 2 | [wenetspeech](wenetspeech/) | [source](https://huggingface.co/datasets/wenet-e2e/wenetspeech) |  |
| CommonVoice (48-lang, Stage 2) | CC0-1.0 | audio | Stage 2 | [commonvoice](commonvoice/) | [source](https://commonvoice.mozilla.org/) |  |
| GigaSpeech | Apache-2.0 | audio | Stage 2 | [gigaspeech](gigaspeech/) | [source](https://huggingface.co/datasets/speechcolab/gigaspeech) |  |
| GigaSpeech 2 | Apache 2.0 | audio | Stage 2 | [gigaspeech2](gigaspeech2/) | [source](https://huggingface.co/datasets/speechcolab/gigaspeech2) |  |
| OmniLingual ASR | CC BY 4.0 | audio | Stage 2 | [omnilingual_asr](omnilingual_asr/) | [source](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) |  |
| EuroSpeech | Mixed | audio | Stage 2 | [eurospeech](eurospeech/) | [source](https://huggingface.co/datasets/disco-eth/EuroSpeech) | [download_permissive.slurm](eurospeech/download_permissive.slurm) - download_permissive.slurm downloads only the 5 countries whose parliamentary license is non-NC/SA (Italy CC-BY-4.0, UK OGL-3.0, Bulgaria CC-BY-2.5, France Licence Ouverte, Norway NLOD-2.0) |
| Granary (YODAS, YTC, VoxPopuli, others) | CC BY 3.0 | audio | Stage 2 | [granary](granary/) | [source](https://huggingface.co/datasets/espnet/yodas-granary) |  |
| ParlaSpeech-RS | CC-BY-4.0 | audio | Stage 2 | [parlaspeech_rs](parlaspeech_rs/) | [source](https://huggingface.co/datasets/classla/ParlaSpeech-RS) |  |
| ParlaMent Parla | CC-BY-4.0 | audio | Stage 2 | [parlament_parla](parlament_parla/) | [source](https://huggingface.co/datasets/projecte-aina/parlament_parla_v3) |  |
| SeamlessAlign | CC-BY-4.0 | audio | Stage 2 | [seamless_align](seamless_align/) | [source](https://huggingface.co/datasets/ai4bharat/SeamlessAlign) |  |
| Kathbath | CC-BY-4.0 | audio | Stage 2 | [kathbath](kathbath/) | [source](https://huggingface.co/datasets/ai4bharat/Kathbath) |  |
| Coral V3 | Openrail | audio | Stage 2 | [coral](coral/) | [source](https://huggingface.co/datasets/CoRal-project/coral-v3) | OpenRAIL license allows commercial use permitted, non-NC/SA, taken as-is, no filtering |
| LegCo Speech | CC-BY-1.0 | audio | Stage 2 | [legco_speech](legco_speech/) | [source](https://huggingface.co/datasets/laubonghaudoi/legco-speech) |  |
| MultiMed | MIT | audio | Stage 2 | [multimed](multimed/) | [source](https://huggingface.co/datasets/leduckhai/MultiMed) |  |
| LibriHeavy | Apache-2.0 | audio | Stage 2 | [libriheavy](libriheavy/) | [source](https://huggingface.co/datasets/mythicinfinity/libriheavy) |  |
| VietSpeech | Apache 2.0 | audio | Stage 2 | [vietspeech](vietspeech/) | [source](https://huggingface.co/datasets/NhutP/VietSpeech) |  |
| Infore2 Audiobooks | CC-BY-4.0 | audio | Stage 2 | [infore2_audiobooks](infore2_audiobooks/) | [source](https://huggingface.co/datasets/doof-ferb/infore2_audiobooks) |  |
| ViMedCSS | CC-BY-4.0 | audio | Stage 2 | [vimedcss](vimedcss/) | [source](https://huggingface.co/datasets/tensorxt/ViMedCSS) |  |
| Kazakh Speech | CC-BY-4.0 | audio | Stage 2 | [kazakh](kazakh/), [kazakh_speech](kazakh_speech/) | [source](https://huggingface.co/datasets/Flamme-VRM/kazakh-speech-dataset) |  |
| MLS | CC-BY-4.0 | audio | Stage 2 | [mls](mls/) | [source](https://huggingface.co/datasets/facebook/multilingual_librispeech) |  |
| F1 Team Radio | CC-BY-4.0 | audio | Stage 2 | [f1_team_radio](f1_team_radio/) | [source](https://huggingface.co/datasets/MikCil/f1-team-radio) |  |
| Zoengjyutgaai | CC-BY-1.0 | audio | Stage 2 | [zoengjyutgaai](zoengjyutgaai/) | [source](https://huggingface.co/datasets/CanCLID/zoengjyutgaai) |  |
| Emilia/YODAS | CC-BY-4.0 | audio | Stage 2 | [Emilia_YODAS](Emilia_YODAS/) | [source](https://huggingface.co/datasets/amphion/Emilia-Dataset) |  |
| HUI-Audio-Corpus-German | CC0-1.0 | audio | Stage 2 | [hui_audio_corpus_german](hui_audio_corpus_german/), [hui_audio_german](hui_audio_german/) | [source](https://opendata.iisys.de/opendata/Datasets/HUI-Audio-Corpus-German) |  |
| Russian LibriSpeech | Public Domain | audio | Cooldown | [russian_librispeech](russian_librispeech/) | [source](https://www.openslr.org/96/) |  |
| SloPalSpeech | CC-BY-4.0 | audio | Cooldown | [slopalspeech](slopalspeech/) | [source](https://huggingface.co/datasets/NaiveNeuron/SloPalSpeech) |  |
| ToneWebinars | Apache 2.0 | audio | Cooldown | [tonewebinars](tonewebinars/) | [source](https://huggingface.co/datasets/Vikhrmodels/ToneWebinars) |  |
| Tuda-De | CC-BY-4.0 | audio | Cooldown | [tuda_de](tuda_de/) | [source](https://huggingface.co/datasets/uhhlt/Tuda-De) |  |
| MediaSpeech | CC-BY-4.0 | audio | Cooldown | [mediaspeech](mediaspeech/), [openslr108_mediaspeech](openslr108_mediaspeech/) | [source](https://www.openslr.org/108/) |  |
| Farsi ASR | MIT | audio | Cooldown | [farsi_asr](farsi_asr/) | [source](https://huggingface.co/datasets/farsi-asr/farsi-asr-dataset) |  |
| Aozora Hurigana | PDM | audio | Cooldown | [aozora_hurigana](aozora_hurigana/) | [source](https://github.com/ndl-lab/hurigana-speech-corpus-aozora) |  |
| Ghana English Speech | CC-BY-4.0, CC-BY-NC-4.0 | audio | Cooldown | [ghana_english](ghana_english/) | [source](https://huggingface.co/datasets/ghananlpcommunity/ghana-english-speech-600hrs) | CC-BY-4.0 at ingestion, relicensed to NC after (inclusion predates the change): [relicense commit](https://huggingface.co/datasets/ghananlpcommunity/ghana-english-speech-600hrs/commit/8f300b3dedb3d90dbe1af3dcd90d3b55aacbf29a) |
| CC-Podcasts | Apache-2.0 | audio | Cooldown | [ccpodcasts](ccpodcasts/) | [source](https://huggingface.co/datasets/shuyuej/CC-Podcasts) |  |
| Localized Narratives (audio) | CC-BY-4.0 | audio | Cooldown | [localized_narratives](localized_narratives/) | [source](https://storage.googleapis.com/localized-narratives) |  |
| MRSAudio | CC-BY-4.0 | audio | Stage 2 | [mrsaudio](mrsaudio/) | [source](https://huggingface.co/datasets/MRSAudio/MRSAudio) |  |
| Marco Longspeech | Apache-2 | audio | SFT | [marco_longspeech](marco_longspeech/) | [source](https://huggingface.co/datasets/AIDC-AI/Marco_Longspeech) |  |
| AudioMCQ StrongAC GeminiCoT | Apache-2 | audio | SFT | [audiomcq_strongac_cot](audiomcq_strongac_cot/) | [source](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT) |  |
| TeleAntiFraud | Apache-2 | audio | SFT | [teleantifraud](teleantifraud/) | [source](https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud) |  |
| VoiceAssistant-400K | Apache-2 | audio | SFT | [voiceassistant_400k](voiceassistant_400k/) | [source](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K) |  |
| HeySQuAD (human + machine) | CC-BY-4.0 | audio | SFT | [heysquad](heysquad/) | [source](https://huggingface.co/datasets/yijingwu/HeySQuAD_human) |  |
| TCM Shizhen Speech (SFT) | Apache-2 | audio | SFT | - | [source](https://huggingface.co/datasets/FreedomIntelligence/TCM-Instruction-Tuning-ShizhenGPT) |  |
| Vocalized SFT | Apache 2.0 | audio | SFT | [vocalized__sft](vocalized__sft/) | [source](https://huggingface.co/datasets/swiss-ai/vocalized-sft) (internal) | in-house generated, Qwen3-TTS vocalization of the already-permissive text-SFT mix (license inherits from the non-NC/SA text sources) |
