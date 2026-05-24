#!/usr/bin/env python3
"""HQ-50K caption generation with Gemma 4 31B base model (completion mode).

Fill-in prompt: image followed by a natural prefix that primes BBC/Nature
documentary style prose. No chat template, no system prompt.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/hq50k_base_caption.py \
        [--n-images 4] [--seed 42]
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
import tarfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HQ50K_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/hf___YangQiee___HQ-50K/downloaded")
MODEL_PATH = "/tmp/models/models--google--gemma-4-31B/snapshots/5dc307ed18be972e945cea1bfd3facf5bfd5fa39"
OUTPUT_DIR = _ROOT / "artifacts" / "hq50k_base_caption"
HQ50K_SHARD_SIZES = [6881, 6834, 6934, 6786, 8410, 961]

# Fill-in prefix — base model completes from here
# <|image|> is the Gemma 4 image placeholder token for completion mode
PROMPT_TEMPLATE = "\t<|image|>\nDescription:"


def load_images(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    shards = sorted(HQ50K_ROOT.glob("*.tar"))
    sizes = HQ50K_SHARD_SIZES[: len(shards)]
    chosen: dict[int, list[int]] = {}
    while sum(len(v) for v in chosen.values()) < n:
        si = rng.choices(range(len(shards)), weights=sizes)[0]
        mi = rng.randrange(sizes[si])
        if mi not in chosen.setdefault(si, []):
            chosen[si].append(mi)
    images = []
    for si, member_indices in sorted(chosen.items()):
        with tarfile.open(shards[si]) as tf:
            jpg_members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
            for mi in member_indices:
                member = jpg_members[mi]
                data = tf.extractfile(member).read()
                images.append(
                    {
                        "key": member.name.replace(".jpg", ""),
                        "shard": shards[si].name,
                        "data": data,
                    }
                )
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from PIL import Image
    from sft_recaption.runtime import configure_worker_environment
    from vllm import LLM, SamplingParams

    configure_worker_environment(0)

    print(f"Loading {args.n_images} HQ-50K images...")
    images = load_images(args.n_images, seed=args.seed)

    print("Loading base model...")
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        download_dir="/tmp/models",
        tensor_parallel_size=1,
        enforce_eager=False,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.80,
        max_num_seqs=4,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1024)

    prompts = []
    for img in images:
        pil_img = Image.open(io.BytesIO(img["data"])).convert("RGB")
        prompts.append(
            {
                "prompt": PROMPT_TEMPLATE,
                "multi_modal_data": {"image": pil_img},
            }
        )

    print(f"\nGenerating {len(prompts)} captions...")
    raw_outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
    outputs = [o.outputs[0].text.strip() for o in raw_outputs]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    for img, caption in zip(images, outputs):
        (OUTPUT_DIR / f"{img['key']}.jpg").write_bytes(img["data"])
        (OUTPUT_DIR / f"{img['key']}.txt").write_text(caption)
        print(f"\n[{img['key']}]\n{caption}\n")

    print(f"{'=' * 60}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
