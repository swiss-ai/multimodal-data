# Smithsonian OCR Augmentation Pass

Runs Qwen3.6-27B (vLLM, no-think mode) over every Smithsonian cleaned5 image,
producing region-grouped OCR transcriptions tagged by text type
(`[handwriting]` / `[printed_label]` / `[printed_article]` / etc.). Output is
written back as Parquet, one file per source shard, with the same `id` column
plus seven OCR columns.

## Browsing the example outputs (with Yazi)

Six paired image + caption files live in [`examples/`](./examples/) covering
the full augmentation spectrum (handwriting, multi-region forms, dense print,
engraving, stamps, NO_TEXT pass-through). Browse them with `yazi`:

```bash
yazi /iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/smithsonian/ocr_pass/examples
```

Each example has a stem-paired pair:

```
example1_handwriting_letter.jpg   ← image (Yazi previews inline)
example1_handwriting_letter.txt   ← curator caption + OCR + augmented caption
```

In Yazi: hover the `.jpg` to see the image in the right pane, press `j` to
move down to the matching `.txt` to read the curator+OCR+augmented sections.
The `.txt` contains exactly what the tokenizer would see for that image.

## Why a single VLM pass and not a dots.mocr filter step

dots.mocr underperforms on:
- handwriting (largest text class in Smithsonian — see tag distribution below)
- degraded printed labels (medicine bottles, faded cardboard, etc.)
- Multi-region layouts (museum cards, broadsides, political cartoons)

Validated on a 100-sample recon: dots.mocr returned 13 chars on a medicine
bottle that contains ~600 chars of legible printed text; Qwen3.6-27B returned
the full transcription with region tags. Using dots.mocr as a pre-filter would
have silently dropped these high-value augmentations. Doing one Qwen pass and
treating output length as the filter signal is the cleaner contract.

## Architecture

- **Serving**: 10 GH200 nodes × 2 vLLM instances/node (TP=2 each) = 20 instances.
  No router. Client distributes work round-robin over all 20 endpoints; vLLM's
  continuous batching handles per-instance concurrency.
- **Discovery**: each SLURM task announces its `<host>:<port>` into
  `${OUTPUT_DIR}/qwen_ocr_endpoints_${JOBID}/${PROCID}.endpoint`. Client reads
  these at startup, health-checks via `/v1/models`, drops instances that fail
  3+ requests.
- **Image preprocessing**: PIL downscale to longest-side 1280px before base64
  encoding. Caps prompt tokens at ~1500 so requests fit comfortably in
  `--max-model-len 16384` even with `max_tokens=1024` reserved for output.
- **Sampling**: `temperature=0`, `repetition_penalty=1.1`,
  `chat_template_kwargs={enable_thinking: False}`. Thinking-off is essential —
  Qwen3.6 burns the entire token budget reasoning if left on (10× slower for
  no quality gain on OCR).

## Throughput (measured, 4/26/2026)

| Setup | req/s aggregate | Wall time for 42K rows |
|---|---|---|
| 1 node × DP=2 TP=2 (1 instance, 4 GPUs) | 0.79 | ~14h |
| 10 nodes × 2 instances, parallelism=80 | **24** | **~30 min** |

## Usage

### 1. Launch the 10-node serving allocation

```bash
unset SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_image \
      SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment \
      SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_workdir
JOBID=$(sbatch --parsable serve_qwen.slurm)
echo "serving job: ${JOBID}"
```

Cold start: ~5 min for all 20 instances to load the 27B model from
`/capstor/store/...` and finish CUDA graph capture. Wait until
`grep -c "Application startup complete" logs/qwen_ocr_serve_${JOBID}.err` ≥ 18.

### 2. Run the OCR client

```bash
python run_ocr.py \
  --endpoints-dir /iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen_ocr_endpoints_${JOBID} \
  --output-dir /capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_full \
  --parallelism 80
```

Resume-safe: existing output parquets are skipped. Parallelism 80 is the
sweet spot for 20 instances (~4 in-flight per instance, well under vLLM's
`max_num_seqs=256`).

### 3. Cancel serving when done

```bash
scancel $JOBID
```

## Output schema (per source shard)

```
id              string         # join key with cleaned5
ocr_elapsed_s   float          # per-image wall time
ocr_chars       int            # length of ocr_text (filter signal)
ocr_no_text     bool           # True if model returned literal "NO_TEXT"
ocr_tags_json   string         # {"handwriting": 2, "printed_label": 3} JSON-serialised
ocr_text        string         # raw transcription with [tag] On the X: markers
ocr_endpoint    string         # which vLLM instance served the request (debug only)
```

## Sample tag distribution (1000-image smoke test)

| Tag | Count |
|---|---|
| printed_article | 458 |
| printed_label | 370 |
| handwriting | **368** |
| engraving | 219 |
| printed_caption | 111 |
| stamp | 23 |
| other | 14 |

Handwriting is the third-largest class — confirms why dots.mocr (which
collapses on cursive) was unsuitable.

## Known caveats

- **Long-tail outliers**: ~1% of images produce >5000 chars due to either
  hallucinated drift (Hutchins genealogy case) or legitimate dense repetition
  (stamp printing sheets). Quarantine via downstream post-filter:
  `df.filter(pl.col("ocr_chars") > 5000)` for manual review before training.
- **Image resize loses fine detail**: 1280px cap means very-small text on
  large posters may degrade. Acceptable trade-off for context-budget safety;
  if needed, raise `--max-model-len 32768` and `MAX_IMG_DIM=2048`.
