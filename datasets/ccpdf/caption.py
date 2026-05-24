#!/usr/bin/env python3
"""
caption.py — Diverse, high-quality captioning of ArXiv figures.

Usage:
  # Local test on sample images:
  python caption.py --mode sample --experiment experiment_01

  # Process a specific shard range (for SLURM array jobs):
  python caption.py --mode shard --shard-start 0 --shard-end 1 --experiment experiment_02

  # Process shards assigned to this SLURM array worker:
  python caption.py --mode slurm --experiment full_run
"""

import argparse
import base64
import io
import json
import os
import random
import struct
import tarfile
from pathlib import Path

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/path/to/data/vision-datasets/processed/hf___mlfoundations___MINT-1T-ArXiv___processed")
SAMPLE_DIR = DATA_ROOT / "sample"
OUTPUT_ROOT = Path("/tmp/toolbox/story_caption/outputs")

GEMMA_PATH = "/tmp/models/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
QWEN_PATH = "/tmp/models/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"

# ---------------------------------------------------------------------------
# Personas & prompt templates
# ---------------------------------------------------------------------------
PERSONAS = [
    "a curious undergraduate student encountering this figure for the first time",
    "an experienced professor explaining this figure to a graduate seminar",
    "a science journalist writing an accessible article for a general audience",
    "a textbook author crafting an explanatory caption for an introductory course",
    "a data analyst focusing on the methodology and statistical aspects",
    "a researcher in the field reviewing this figure in a peer-review context",
    "a technical writer documenting this figure for a survey paper",
    "a PhD candidate summarizing this figure for their literature review",
]

PROMPT_STYLES = [
    {
        "name": "interpret",
        "instruction": (
            "Interpret the figure: explain what it shows, what the key findings or "
            "patterns are, and what conclusions can be drawn. Focus on the scientific "
            "meaning and significance."
        ),
    },
    {
        "name": "describe",
        "instruction": (
            "Describe the figure in detail: what type of visualization it is, what the "
            "axes represent, what data is plotted, any labels, legends, or annotations "
            "present. Be precise and thorough about visual elements."
        ),
    },
    {
        "name": "analyze",
        "instruction": (
            "Analyze the figure: discuss the trends, distributions, or relationships "
            "visible in the data. Note any outliers, patterns, or notable features. "
            "Comment on the quality and clarity of the presentation."
        ),
    },
    {
        "name": "educate",
        "instruction": (
            "Explain this figure as if teaching someone unfamiliar with the field. "
            "Provide context for what is being measured and why, and guide the reader "
            "through understanding the key takeaways."
        ),
    },
]

CAPTION_PROMPT = """\
You are {persona}.

Look at this scientific figure from an academic paper and {instruction}

Guidelines:
- Only describe what you can actually see in the image. Do not invent data points, \
values, or details that are not visible.
- If you recognize the type of plot, measurement, or scientific domain from visual \
cues (axis labels, curve shapes, distributions), you may use your background \
knowledge to provide context — but only if you are confident.
- Adapt the length of your response to the complexity of the figure: simple figures \
get shorter captions, complex multi-panel figures get longer ones.
- Do not start with "This image shows" or "The figure depicts" — jump straight into \
the content.
- Do not mention that you are an AI or that you are looking at an image.
- Write in flowing prose, not bullet points."""

FILTER_PROMPT = """\
Look at this image from a scientific paper. Assess whether it contains enough \
interpretable visual content to generate a meaningful caption.

A figure is WORTH CAPTIONING if it has ANY of these:
- Plots or charts with readable axes, data points, curves, or trends
- Diagrams with labeled components or clear structure
- Photographs or microscopy images showing identifiable objects or phenomena
- Tables with readable content
- Schematics or flowcharts with understandable elements

A figure should be SKIPPED if:
- It is mostly blank, corrupted, or too low-resolution to read
- It contains only unlabeled abstract patterns with no text or axes
- It is a simple geometric shape or logo with no scientific content
- It is a screenshot of code or a plain text block

Respond with EXACTLY one of:
CAPTION — if the figure is worth captioning
SKIP — if the figure should be skipped

Your response (CAPTION or SKIP):"""


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """Return (width, height) from JPEG bytes without PIL."""
    f = io.BytesIO(data)
    f.read(2)  # skip SOI
    while True:
        marker, size = struct.unpack(">HH", f.read(4))
        if 0xFFC0 <= marker <= 0xFFC3:  # SOF markers
            f.read(1)  # precision
            h, w = struct.unpack(">HH", f.read(4))
            return w, h
        f.read(size - 2)


def model_short_name(model_path: str) -> str:
    """Extract a readable model name from the path."""
    # e.g. 'models--google--gemma-4-31B-it' -> 'gemma-4-31B-it'
    for part in model_path.split("/"):
        if part.startswith("models--"):
            return part.split("--", 2)[-1]
    return model_path


def to_data_url(jpg_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(jpg_bytes).decode('ascii')}"


def build_filter_messages(jpg_bytes: bytes) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_data_url(jpg_bytes)}},
                {"type": "text", "text": FILTER_PROMPT},
            ],
        }
    ]


def build_caption_messages(jpg_bytes: bytes, persona: str, style: dict) -> list[dict]:
    prompt = CAPTION_PROMPT.format(persona=persona, instruction=style["instruction"])
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_data_url(jpg_bytes)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_images_from_sample_dir(limit: int = 0) -> list[dict]:
    """Load images from the sample/ directory."""
    files = sorted(SAMPLE_DIR.glob("*.jpg"))
    if limit:
        files = files[:limit]
    return [{"key": f.stem, "jpg": f.read_bytes()} for f in files]


def iter_shard_chunks(shard_idx: int, chunk_size: int = 2000, limit: int = 0):
    """Yield chunks of images from a tar shard to avoid loading all into memory."""
    shard_path = DATA_ROOT / f"shard_{shard_idx:03d}.tar"
    if not shard_path.exists():
        print(f"  Shard {shard_path} not found, skipping.")
        return
    chunk = []
    total = 0
    with tarfile.open(shard_path, "r") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".jpg"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            key = member.name.rsplit(".", 1)[0]
            chunk.append({"key": key, "jpg": f.read()})
            total += 1
            if limit and total >= limit:
                yield chunk
                return
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def process_chunk(
    samples: list[dict],
    llm: LLM,
    chat_kwargs: dict,
    rng: random.Random,
    out_dir: Path,
    batch_size: int,
    model_path: str,
) -> tuple[int, int]:
    """Filter and caption a chunk of images. Returns (accepted, skipped) counts."""
    filter_sampling = SamplingParams(temperature=0.0, max_tokens=16)
    caption_sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)
    model_name = model_short_name(model_path)

    # --- Filter ---
    accepted = []
    skipped = 0
    for batch_start in range(0, len(samples), batch_size):
        batch = samples[batch_start : batch_start + batch_size]
        convos = [build_filter_messages(s["jpg"]) for s in batch]
        outputs = llm.chat(convos, sampling_params=filter_sampling, use_tqdm=True, **chat_kwargs)
        for s, out in zip(batch, outputs):
            verdict = out.outputs[0].text.strip().upper()
            if "CAPTION" in verdict:
                accepted.append(s)
            else:
                skipped += 1

    if not accepted:
        return 0, skipped

    # --- Caption ---
    for s in accepted:
        s["persona"] = rng.choice(PERSONAS)
        s["style"] = rng.choice(PROMPT_STYLES)

    for batch_start in range(0, len(accepted), batch_size):
        batch = accepted[batch_start : batch_start + batch_size]
        convos = [build_caption_messages(s["jpg"], s["persona"], s["style"]) for s in batch]
        outputs = llm.chat(convos, sampling_params=caption_sampling, use_tqdm=True, **chat_kwargs)
        for s, out in zip(batch, outputs):
            s["caption"] = out.outputs[0].text.strip()

    # --- Write ---
    for s in accepted:
        safe_key = s["key"].replace("/", "_")
        (out_dir / f"{safe_key}.jpg").write_bytes(s["jpg"])
        (out_dir / f"{safe_key}.txt").write_text(s["caption"], encoding="utf-8")
        try:
            w, h = jpeg_dimensions(s["jpg"])
        except Exception:
            w, h = -1, -1
        meta = {
            "key": s["key"],
            "model": model_name,
            "persona": s["persona"],
            "style": s["style"]["name"],
            "caption_chars": len(s["caption"]),
            "image_width": w,
            "image_height": h,
            "image_bytes": len(s["jpg"]),
        }
        (out_dir / f"{safe_key}.json").write_text(json.dumps(meta))

    return len(accepted), skipped


def run_on_shard(
    shard_idx: int,
    model_path: str,
    out_dir: Path,
    batch_size: int,
    seed: int,
    limit: int = 0,
    llm: LLM | None = None,
    chat_kwargs: dict | None = None,
):
    """Process a single shard, streaming chunks to keep memory bounded."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    is_qwen = "Qwen" in model_path

    # Load model if not provided (for sample/shard mode)
    if llm is None:
        print(f"Loading model from {model_path} ...")
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.80,
            dtype="bfloat16",
            max_model_len=8192,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )
    if chat_kwargs is None:
        chat_kwargs = {}
        if is_qwen:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    total_accepted = 0
    total_skipped = 0

    for chunk_i, chunk in enumerate(iter_shard_chunks(shard_idx, chunk_size=2000, limit=limit)):
        print(f"\n  Shard {shard_idx:03d} chunk {chunk_i}: {len(chunk)} images")
        accepted, skipped = process_chunk(chunk, llm, chat_kwargs, rng, out_dir, batch_size, model_path)
        total_accepted += accepted
        total_skipped += skipped
        print(f"    Accepted: {accepted}, Skipped: {skipped}")

    # Write metadata
    meta = {
        "model": model_path.split("/")[-3] if "/" in model_path else model_path,
        "shard": shard_idx,
        "accepted": total_accepted,
        "skipped": total_skipped,
        "seed": seed,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Shard {shard_idx:03d} done: {total_accepted} accepted, {total_skipped} skipped")
    return llm, chat_kwargs


def run_on_samples(
    model_path: str,
    out_dir: Path,
    batch_size: int,
    seed: int,
    limit: int = 0,
):
    """Process sample directory images (small set, fits in memory)."""
    samples = load_images_from_sample_dir(limit=limit)
    if not samples:
        print("No sample images found.")
        return

    rng = random.Random(seed)
    is_qwen = "Qwen" in model_path

    print(f"Loading model from {model_path} ...")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.80,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    chat_kwargs = {}
    if is_qwen:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    out_dir.mkdir(parents=True, exist_ok=True)
    accepted, skipped = process_chunk(samples, llm, chat_kwargs, rng, out_dir, batch_size, model_path)

    meta = {
        "model": model_path.split("/")[-3] if "/" in model_path else model_path,
        "total_input": len(samples),
        "accepted": accepted,
        "skipped": skipped,
        "seed": seed,
    }
    (out_dir / "_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. {accepted} accepted, {skipped} skipped -> {out_dir}")

    # Print first few for inspection
    txts = sorted(out_dir.glob("*.txt"))[:5]
    for t in txts:
        if t.name.startswith("_"):
            continue
        print(f"\n{'=' * 60}")
        print(f"Key: {t.stem}")
        caption = t.read_text()
        print(f"Caption ({len(caption)}c): {caption[:300]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sample", "shard", "slurm"],
        required=True,
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="limit images per source")
    parser.add_argument("--model", default="gemma", choices=["gemma", "qwen", "auto"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / args.experiment

    def pick_model(shard_idx: int) -> str:
        """95% Gemma, 5% Qwen (every 20th shard)."""
        if args.model == "auto":
            return QWEN_PATH if shard_idx % 20 == 0 else GEMMA_PATH
        return GEMMA_PATH if args.model == "gemma" else QWEN_PATH

    if args.mode == "sample":
        model_path = GEMMA_PATH if args.model != "qwen" else QWEN_PATH
        run_on_samples(model_path, out_dir, args.batch_size, args.seed, args.limit)

    elif args.mode == "shard":
        for shard_idx in range(args.shard_start, args.shard_end):
            model_path = pick_model(shard_idx)
            shard_out = out_dir / f"shard_{shard_idx:03d}"
            run_on_shard(shard_idx, model_path, shard_out, args.batch_size, args.seed, args.limit)

    elif args.mode == "slurm":
        worker_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        num_workers = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "128"))
        total_shards = 128

        my_shards = [s for s in range(total_shards) if s % num_workers == worker_id]
        print(f"Worker {worker_id}/{num_workers}: processing shards {my_shards}")

        # Group by model to avoid reloading
        llm = None
        chat_kwargs = None
        current_model = None
        for shard_idx in my_shards:
            model_path = pick_model(shard_idx)
            if model_path != current_model:
                # Need to load a different model
                del llm  # free GPU memory
                llm = None
                current_model = model_path
            shard_out = out_dir / f"shard_{shard_idx:03d}"
            llm, chat_kwargs = run_on_shard(
                shard_idx,
                model_path,
                shard_out,
                args.batch_size,
                args.seed,
                args.limit,
                llm,
                chat_kwargs,
            )


if __name__ == "__main__":
    main()
