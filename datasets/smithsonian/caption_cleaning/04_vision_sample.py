#!/usr/bin/env python3
"""
04_vision_sample.py

Vision-grounded recaptioning test.
For each of the 8 approved subsets, sample 4 images and generate
grounded descriptions using Gemma 4 31B-it (multimodal).

Outputs flat WDS tars to OUT_ROOT (smithsonian_cleaned3/).
One tar per subset: t1_nmaahc.tar, t1_npg.tar, etc.
"""

import base64
import io
import random
import tarfile
from pathlib import Path

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WDS_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian")
OUT_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian/smithsonian_cleaned3")

MODEL_PATH = "/tmp/models/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.80
MAX_TOKENS = 1024
TEMPERATURE = 0.2
TOP_P = 0.9
SAMPLES_PER_SUBSET = 4
RANDOM_SEED = 42

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
# Helpers
# ---------------------------------------------------------------------------


def subset_to_flat(subset_key: str) -> str:
    """'tier1/nmaahc' -> 't1_nmaahc', 'tier2/design/chndm' -> 't2_design_chndm'"""
    return subset_key.replace("tier", "t", 1).replace("/", "_")


def to_data_url(jpg_bytes: bytes) -> str:
    encoded = base64.b64encode(jpg_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def sample_from_subset(subset_dir: Path, n: int, rng: random.Random) -> list[dict]:
    """
    Sample up to n (jpg, txt, json) triplets from tars in subset_dir.
    Returns list of dicts: {key, jpg, txt, json_bytes}.
    """
    tars = sorted(subset_dir.glob("*.tar"))
    if not tars:
        return []

    # Spread across shards
    indices = [int(i * len(tars) / n) for i in range(min(n, len(tars)))]
    chosen_tars = [tars[i] for i in indices]
    rng.shuffle(chosen_tars)

    samples = []
    seen_keys = set()

    for tar_path in chosen_tars:
        if len(samples) >= n:
            break
        try:
            with tarfile.open(tar_path, "r") as tf:
                buffers: dict[str, dict] = {}
                for member in tf:
                    if not member.isfile():
                        continue
                    if "." not in member.name:
                        continue
                    stem, ext = member.name.rsplit(".", 1)
                    if ext not in ("jpg", "txt", "json"):
                        continue
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

                candidates = [
                    {"key": k, **v}
                    for k, v in buffers.items()
                    if "jpg" in v
                    and "txt" in v
                    and "json_bytes" in v
                    and k not in seen_keys
                    and len(v.get("txt", "")) >= 100
                ]
                if not candidates:
                    continue
                pick = rng.choice(candidates)
                samples.append(pick)
                seen_keys.add(pick["key"])
        except Exception as exc:
            print(f"  WARNING: {tar_path}: {exc}")

    return samples[:n]


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


def write_tar(out_path: Path, samples: list[dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w") as tf:
        for s in samples:
            for ext, content in [
                ("jpg", s["jpg"]),
                ("txt", s["recaption"].encode("utf-8")),
                ("json", s["json_bytes"]),
            ]:
                info = tarfile.TarInfo(name=f"{s['key']}.{ext}")
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(RANDOM_SEED)

    print("Sampling images from approved subsets ...")
    all_samples: list[dict] = []  # each dict gets a 'subset_key' field added
    for subset_key in APPROVED_SUBSETS:
        subset_dir = WDS_ROOT / subset_key
        picked = sample_from_subset(subset_dir, SAMPLES_PER_SUBSET, rng)
        for s in picked:
            s["subset_key"] = subset_key
        all_samples.extend(picked)
        print(f"  {subset_key}: {len(picked)} samples")

    print(f"\nTotal: {len(all_samples)} samples. Loading model ...")

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

    print(f"Running vision recaptioning on {len(all_samples)} samples ...")
    conversations = [build_messages(s["jpg"], s["txt"]) for s in all_samples]
    outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=True)

    for s, out in zip(all_samples, outputs):
        s["recaption"] = out.outputs[0].text.strip()

    # Print before/after for inspection
    sep = "-" * 60
    for s in all_samples:
        print(f"\n{sep}")
        print(f"[{s['subset_key']}]  {s['key']}")
        print(f"RAW ({len(s['txt'])}c): {s['txt'][:300]}")
        print(f"NEW ({len(s['recaption'])}c): {s['recaption'][:400]}")

    # Write one tar per subset
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    by_subset: dict[str, list[dict]] = {}
    for s in all_samples:
        by_subset.setdefault(s["subset_key"], []).append(s)

    for subset_key, samples in by_subset.items():
        flat_name = subset_to_flat(subset_key)
        out_path = OUT_ROOT / f"{flat_name}.tar"
        write_tar(out_path, samples)
        print(f"\n-> {out_path}  ({len(samples)} samples)")

    print("\nDone.")


if __name__ == "__main__":
    main()
