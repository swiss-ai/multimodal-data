#!/usr/bin/env python3
"""HQ-50K caption generation with DOCCI few-shot examples.

For each image, randomly samples 16 DOCCI captions as examples and
generates one plain-text caption. No JSON, no format rules.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/hq50k_docci_fewshot.py [--n-images 16]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tarfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DOCCI_ARROW = Path("/path/to/data/vision-datasets/raw/stage2/hf___google___docci/docci-train.arrow")
HQ50K_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/hf___YangQiee___HQ-50K/downloaded")
MODEL_PATH = os.environ.get("RECAPTION_MODEL_PATH", "")
OUTPUT_DIR = _ROOT / "artifacts" / "hq50k_docci_fewshot_img"

DOCCI_IMAGES_DIR = DOCCI_ARROW.parent / "images"

# Known shard sizes — avoids scanning all tar members
HQ50K_SHARD_SIZES = [6881, 6834, 6934, 6786, 8410, 961]

N_EXAMPLES_PER_IMAGE = 16  # image+caption pairs; 17 images total


def load_docci_index() -> list[dict]:
    """Load DOCCI captions + paths only — no image bytes loaded upfront."""
    import pyarrow.ipc as ipc

    with ipc.open_stream(DOCCI_ARROW) as f:
        table = f.read_all()
    index = []
    for i in range(len(table)):
        example_id = table["example_id"][i].as_py()
        img_path = DOCCI_IMAGES_DIR / f"{example_id}.jpg"
        if not img_path.exists():
            continue
        index.append(
            {
                "img_path": img_path,
                "caption": table["description"][i].as_py(),
            }
        )
    return index


def load_images(n: int, seed: int = 42) -> list[dict]:
    """Sample n images from HQ-50K, grouping by shard to minimise tar opens."""
    rng = random.Random(seed)
    shards = sorted(HQ50K_ROOT.glob("*.tar"))
    sizes = HQ50K_SHARD_SIZES[: len(shards)]
    total = sum(sizes)

    # Pick n (shard, position) pairs weighted by shard size
    chosen: dict[int, list[int]] = {}
    for _ in range(min(n, total)):
        while True:
            si = rng.choices(range(len(shards)), weights=sizes)[0]
            mi = rng.randrange(sizes[si])
            if mi not in chosen.setdefault(si, []):
                chosen[si].append(mi)
                break

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
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading DOCCI index (captions + paths)...")
    all_docci = load_docci_index()
    print(f"  {len(all_docci)} examples indexed")

    print(f"Loading {args.n_images} random HQ-50K images...")
    images = load_images(args.n_images, seed=args.seed)
    print(f"  Loaded {len(images)} images")

    from sft_recaption.config import ModelConfig
    from sft_recaption.runtime import (
        VLLMChatEngine,
        configure_worker_environment,
        to_data_url,
    )
    from sft_recaption.schemas import ImagePayload

    configure_worker_environment(0)
    engine = VLLMChatEngine(
        ModelConfig(
            model_repo=MODEL_PATH,
            tensor_parallel_size=1,
            max_num_seqs=4,
            enforce_eager=False,
            limit_mm_per_prompt={"image": 17},  # 16 docci + 1 target
        )
    )

    # Build multi-turn conversations: [img, caption] pairs then target image
    rng = random.Random(args.seed + 1)
    conversations = []
    for img in images:
        examples = rng.sample(all_docci, N_EXAMPLES_PER_IMAGE)
        target_payload = ImagePayload(media_type="image/jpeg", data=img["data"])

        messages = []
        for ex in examples:
            ex_payload = ImagePayload(media_type="image/jpeg", data=ex["img_path"].read_bytes())
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": to_data_url(ex_payload)},
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": ex["caption"],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": to_data_url(target_payload)},
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        )
        conversations.append(messages)

    print(f"\nGenerating {len(conversations)} captions...")
    outputs = engine.chat(conversations, temperature=0.7, top_p=0.9, max_tokens=1024)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    for img, raw in zip(images, outputs):
        caption = raw.strip()

        # Save caption
        out_path = OUTPUT_DIR / f"{img['key']}.txt"
        with out_path.open("w") as f:
            f.write(f"KEY: {img['key']} ({img['shard']})\n\n")
            f.write(caption)

        # Save image
        img_path = OUTPUT_DIR / f"{img['key']}.jpg"
        img_path.write_bytes(img["data"])

        print(f"\n[{img['key']}]")
        print(caption)
        print()

    print(f"{'=' * 60}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
