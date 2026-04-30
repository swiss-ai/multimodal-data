# ShadenA/MathNet

Multimodal olympiad math corpus. Problem statements (LaTeX markdown + diagrams)
paired with one or more expert-written solutions, covering ~60 countries and
17 languages.

- Source: https://huggingface.co/datasets/ShadenA/MathNet
- License: **CC-BY-4.0** — permissive for commercial use; attribution required.
- Size: 27.8K rows.
- Format: parquet (image_bytes inline), 59 subsets (per-competition splits).
- Paper: arXiv 2604.18584

## Row schema

```
id:                 unique problem identifier
problem_markdown:   LaTeX problem statement (may embed images via ![](...))
solutions_markdown: list of solutions, each LaTeX markdown (may embed figures)
images:             list of {name, bytes} — problem + solution figures
country:            origin country (e.g. "Argentina", "China")
competition:        contest name (IMO, APMO, USAMO, ...)
topics_flat:        list of topic tags (combinatorics, geometry, ...)
language:           ISO code of the problem language
problem_type:       "proof only" OR "proof and answer"
final_answer:       expected final answer (empty for pure proof problems)
```

## Training role

SFT-scale dataset (~50M tokens), ideal for:

1. **Stage 3 SFT — math reasoning**: canonical placement. Mix with GSM8K,
   MATH, NuminaMath, OpenMathInstruct for breadth.
2. **Stage 2 visual instruction pretraining — math-diagram exposure**:
   thin slice (~1-3% of stage-2 volume) to expose the model to math
   figures early. Too small to bias full pretraining (~0.05% of a 100B
   pretraining budget), but uniquely valuable because most math SFT
   corpora are text-only.

Note: `solutions_markdown` can contain `![](attached_image_N.png)` tags
pointing to solution-side figures (not just problem figures). For a standard
text-only VLM (vision in, text out), strip these tags before SFT. For a
unified generative VLM (interleaved image+text output), keep them.

## Formatting suggestion

```python
# For "proof and answer" problems with thinking-mode VLM:
{"user": problem_markdown,
 "assistant": f"<think>{solution}</think>{final_answer}"}

# For "proof only" problems:
{"user": problem_markdown,
 "assistant": solution}  # no thinking wrapper; proof IS the answer
```

## Download

```bash
sbatch /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/mathnet/download.slurm
```

Expected wall time: ~5-15 min (27.8K rows, few GB parquet with embedded
images, fast single-host hf_transfer).

Output location: `/capstor/store/cscs/swissai/infra01/vision-datasets/raw/mathnet/`
