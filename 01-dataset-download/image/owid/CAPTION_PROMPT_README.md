# OWID Caption Prompt — Production (V12)

**Prompt file**: `CAPTION_PROMPT_V12_OWID_PRODUCTION.txt`

**Use**: generating data-story captions for the 3,487 OWID standalone grapher charts (`processed/owid___charts/grapher_charts_standalone.parquet`).

## What V12 does

- **Output**: `title\n\nsubtitle\n\n200-300 word data-story paragraph`.
- **Metadata-aware**: title / subtitle / note are passed as ground truth in the user prompt and used verbatim; the chart image is only used for pattern/trend description. This eliminates OCR-style hallucinations of note text.
- **Zero-hallucination on regions**: for choropleth maps, only the darkest clearly-separated countries get specific band values; ambiguous regions are described qualitatively ("lighter shades", "mid-range"). Avoids "whole continent is 0-10" bias.
- **Generic cluster rule**: 3+ visual elements in the same band are described as a group, not assigned fabricated per-element values.
- **Excludes page chrome**: no "Data source:", no OurWorldinData URL, no "CC BY".
- **Neutral voice**: no moral framing, no hedges, no soft-causal verbs.

## Run

```
python3 prompt_iter/run_prompt.py \
    --prompt-file CAPTION_PROMPT_V12_OWID_PRODUCTION.txt \
    --tag prod \
    --metadata-aware \
    --temperature 0.1
```

Iteration history (for reference): V1→V12 in `prompt_iter/prompt_v*.txt` with 100-sample outputs in `prompt_iter/out_v*_100/` and comparison recons in `prompt_iter/RECON_*.md`.

## Iteration summary

| ver | wc mean | banned | note_cap | main change |
|-----|---------|--------|----------|-------------|
| V1  | 158     | 10     | 0/45     | baseline data-story |
| V4  | 158     | 9      | 0/45     | tighter hedge ban + consistent-extrema |
| V8  | 274     | 26     | 29/45    | title+subtitle layout + notes from pixels |
| V10 | 239     | 38     | 25/45    | short prompt + metadata-aware (notes injected as ground truth) |
| V12 | 251     | 35     | 27/45    | **production**: zero-hallucination regional claims |
