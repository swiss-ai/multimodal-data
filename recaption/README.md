# Recaption

Regenerate captions for image datasets with open-weight VLMs, and generate
structured SFT data. Each engine lives in its own subdirectory.

## Engines

| Directory | Purpose |
|-----------|---------|
| `vllm/` | Streaming batch recaption with vLLM (Qwen and others). One loader per dataset plus consolidation to webdataset, driven by SLURM array jobs (`run.slurm`) |
| `blip3/` | Multi-GPU BLIP3 recaption: chunk, recaption, merge, validate |
| `sft/`, `sft2/` | Structured SFT generation: a VLM produces reasoning QA, a second VLM judges groundedness and quality (HQ-50K, Google RSRCC, DRiM-VisualReason-Hard, MINT-1T-ArXiv) |

## Shared top-level

- `base.py` Engine interface shared by the recaption scripts.
- `instruct.py`, `qinstruct.py`, `thinking.py` vLLM batch caption-generation variants (one SLURM task per tar).
- `main.py`, `wds.py` Entry point and webdataset helpers.
- `prompts/` Prompt templates.

## Adding a dataset

Under `vllm/`, add a `loader_<dataset>.py` that defines the data source,
sampling, prompt template, and model config, following the existing loaders.
Select it at submit time:

```bash
RECAPTION_LOADER=loader_recap_datacomp_1b_downloaded_v2 sbatch --array=0-99 recaption/vllm/run.slurm
```
