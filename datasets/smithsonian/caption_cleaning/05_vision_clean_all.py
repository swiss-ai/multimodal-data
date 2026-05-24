#!/usr/bin/env python3
"""
05_vision_clean_all.py

Distributed vision-grounded Smithsonian recaptioner. Run as a SLURM array job.

For each assigned tar shard (round-robin by global tar index % num_workers):
  1. Read all samples (.jpg + .txt + .json triplets).
  2. Run multimodal LLM (Gemma 4 31B-it) with image + raw caption → grounded description.
  3. Post-filter: skip samples with len(new_caption) < MIN_LEN.
  4. Write a new flat output tar at OUT_ROOT/{flat_name} containing only kept
     samples: original .jpg + .json, new .txt.

Usage:
    python 05_vision_clean_all.py --worker-id N --num-workers M
"""

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WDS_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian")
OUT_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian/smithsonian_cleaned4")
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "vision_clean_checkpoints4"

MODEL_PATH = "/tmp/models/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.80
MAX_TOKENS = 1024
TEMPERATURE = 0.2
TOP_P = 0.9
MIN_LEN = 200  # post-filter: skip outputs shorter than this

APPROVED_SUBSETS = [
    "tier1/nmaahc",
    "tier1/npg",
    "tier2/design/chndm",
    "tier2/history/nasm",
    "tier2/history/nmah",
    "tier2/other/acm",
    "tier2/other/npm",
    "tier2/other/sia",
]

VISION_PROMPT = """\
You are a museum caption editor. Lightly refine the original catalog record below, \
using the image as a reference.

The original record is the primary source — preserve its wording and detail as much \
as possible. Only make changes where necessary:

1. Fix formatting: broken line breaks, run-on or abruptly cut sentences, obvious OCR \
artifacts (e.g. "photog- raphy" → "photography").
2. Remove content that is irrelevant or unsuitable: bare dimension blocks \
("H x W: …"), accession numbers, pipe-separated metadata fields, provenance lines \
("From National Portrait Gallery.", "Place of origin: …"), display-status notes, \
condition codes, creator/attribution lines.
3. If the remaining text already reads as natural descriptive prose, preserve it \
as-is. If it does not — whether it is a title, a short phrase, or a longer \
structured caption — write a descriptive paragraph using the image as the primary \
source: describe what you observe visually, and use the original record only for \
context such as the object name, date, or event.
4. Do not rewrite, paraphrase, or summarise. Keep the author's voice and all \
descriptive detail.
5. Do not open with "The image shows", "This image depicts", or similar meta-phrases.
6. Output ONLY the cleaned text, nothing else.

Original record:
{caption}

Cleaned:"""


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def subset_and_tar_to_flat(subset_key: str, tar_name: str) -> str:
    """'tier1/nmaahc', '000000.tar' -> 't1_nmaahc_000000.tar'"""
    flat_prefix = subset_key.replace("tier", "t", 1).replace("/", "_")
    return f"{flat_prefix}_{tar_name}"


# ---------------------------------------------------------------------------
# Tar I/O
# ---------------------------------------------------------------------------


def discover_tars() -> list[tuple[str, Path]]:
    """Return sorted list of (subset_key, tar_path) for all approved subsets."""
    result = []
    for subset in APPROVED_SUBSETS:
        d = WDS_ROOT / subset
        for tar in sorted(d.glob("*.tar")):
            result.append((subset, tar))
    return result


def read_tar_samples(tar_path: Path) -> list[dict]:
    """
    Stream through a tar once, collecting all (jpg, txt, json) triplets.
    Returns list of dicts: {key, jpg, txt, json_bytes}.
    """
    buffers: dict[str, dict] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if "." not in name:
                continue
            stem, ext = name.rsplit(".", 1)
            f = tf.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            if stem not in buffers:
                buffers[stem] = {}
            if ext == "jpg":
                buffers[stem]["jpg"] = raw
            elif ext == "txt":
                buffers[stem]["txt"] = raw.decode("utf-8", errors="replace").strip()
            elif ext == "json":
                buffers[stem]["json_bytes"] = raw

    samples = []
    for key, parts in buffers.items():
        if "jpg" in parts and "txt" in parts and "json_bytes" in parts:
            samples.append({"key": key, **parts})
    return samples


def write_output_tar(out_path: Path, kept: list[dict]) -> None:
    """Write kept samples to a new tar. Order: jpg, txt, json."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w") as tf:
        for s in kept:
            for ext, content in [
                ("jpg", s["jpg"]),
                ("txt", s["new_txt"].encode("utf-8")),
                ("json", s["json_bytes"]),
            ]:
                info = tarfile.TarInfo(name=f"{s['key']}.{ext}")
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> set[str]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"Checkpoint loaded: {len(data)} tars already done")
        return set(data)
    return set()


def save_checkpoint(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(completed), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def to_data_url(jpg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_messages(jpg_bytes: bytes, caption: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_data_url(jpg_bytes)}},
                {"type": "text", "text": VISION_PROMPT.format(caption=caption)},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()

    worker_id = args.worker_id
    num_workers = args.num_workers

    checkpoint_path = CHECKPOINT_DIR / f"worker_{worker_id:03d}.json"
    completed = load_checkpoint(checkpoint_path)

    all_tars = discover_tars()
    my_tars = [(s, p) for i, (s, p) in enumerate(all_tars) if i % num_workers == worker_id]
    pending = [(s, p) for s, p in my_tars if str(p) not in completed]

    print(
        f"[worker {worker_id}/{num_workers}] assigned {len(my_tars)} tars, {len(pending)} pending",
        flush=True,
    )
    if not pending:
        print("Nothing to do.")
        return

    print(f"[worker {worker_id}] Loading model ...", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)

    total_read = total_llm = total_post_filtered = total_kept = 0

    for tar_idx, (subset_key, tar_path) in enumerate(pending, 1):
        print(
            f"[worker {worker_id}] [{tar_idx}/{len(pending)}] {subset_key}/{tar_path.name}",
            flush=True,
        )
        flat_name = subset_and_tar_to_flat(subset_key, tar_path.name)
        out_path = OUT_ROOT / flat_name

        # --- Read ---
        samples = read_tar_samples(tar_path)
        total_read += len(samples)
        total_llm += len(samples)
        print(f"  {len(samples)} samples -> LLM", flush=True)

        # --- LLM ---
        kept = []
        if samples:
            conversations = [build_messages(s["jpg"], s["txt"]) for s in samples]
            outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=True)
            n_post = 0
            for s, out in zip(samples, outputs):
                new_txt = out.outputs[0].text.strip()
                if len(new_txt) < MIN_LEN:
                    n_post += 1
                else:
                    kept.append(
                        {
                            "key": s["key"],
                            "jpg": s["jpg"],
                            "new_txt": new_txt,
                            "json_bytes": s["json_bytes"],
                        }
                    )
            total_post_filtered += n_post
            print(f"  kept {len(kept)}, post-filtered {n_post}", flush=True)

        total_kept += len(kept)

        # --- Write ---
        if kept:
            write_output_tar(out_path, kept)
            print(f"  -> {out_path} ({len(kept)} samples)", flush=True)
        else:
            print("  -> no samples kept, tar skipped", flush=True)

        completed.add(str(tar_path))
        save_checkpoint(checkpoint_path, completed)

    print(f"\n[worker {worker_id}] === Done ===")
    print(f"  Read:          {total_read}")
    print(f"  LLM calls:     {total_llm}")
    print(f"  Post-filtered: {total_post_filtered}  (new_txt < {MIN_LEN}c)")
    print(f"  Kept:          {total_kept}")


if __name__ == "__main__":
    main()
