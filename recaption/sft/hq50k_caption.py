#!/usr/bin/env python3
"""HQ-50K caption generation — clean single prompt, no few-shot.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/hq50k_caption.py [--n-images 16] [--seed 42]
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

HQ50K_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/hf___YangQiee___HQ-50K/downloaded")
DOCCI_ARROW = Path("/path/to/data/vision-datasets/raw/stage2/hf___google___docci/docci-train.arrow")
DOCCI_IMAGES_DIR = DOCCI_ARROW.parent / "images"
MODEL_PATH = "/tmp/models/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
OUTPUT_DIR = _ROOT / "artifacts" / "hq50k_caption"

HQ50K_SHARD_SIZES = [6881, 6834, 6934, 6786, 8410, 961]
N_DOCCI_EXAMPLES = 10

PROMPT = """\
Describe this image accurately and in detail. Cover all visible elements: \
subjects, objects, setting, colors, materials, text, and spatial relationships. \
Write in plain, factual prose — no poetic flourishes, no inferred emotions or \
atmosphere, no camera or lens language. Do not open with "The image shows", \
"In this photograph", or any phrase referencing the medium. \
Describe only what is directly visible.\
"""


def load_docci_index() -> list[dict]:
    import pyarrow.ipc as ipc

    with ipc.open_stream(DOCCI_ARROW) as f:
        table = f.read_all()
    index = []
    for i in range(len(table)):
        example_id = table["example_id"][i].as_py()
        img_path = DOCCI_IMAGES_DIR / f"{example_id}.jpg"
        if img_path.exists():
            index.append({"img_path": img_path, "caption": table["description"][i].as_py()})
    return index


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
    parser.add_argument("--n-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading DOCCI index...")
    docci = load_docci_index()
    print(f"  {len(docci)} examples")

    print(f"Loading {args.n_images} HQ-50K images...")
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
            download_dir=Path("/tmp/models"),
            limit_mm_per_prompt={"image": N_DOCCI_EXAMPLES + 1},
        )
    )

    rng = random.Random(args.seed + 1)
    conversations = []
    for img in images:
        examples = rng.sample(docci, N_DOCCI_EXAMPLES)
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
                        {"type": "text", "text": PROMPT},
                    ],
                }
            )
            messages.append({"role": "assistant", "content": ex["caption"]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": to_data_url(target_payload)},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        )
        conversations.append(messages)

    print(f"\nGenerating {len(conversations)} captions...")
    outputs = engine.chat(conversations, temperature=0.7, top_p=0.9, max_tokens=1024)

    import shutil

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print(f"\n{'=' * 60}")
    for img, raw in zip(images, outputs):
        caption = raw.strip()
        (OUTPUT_DIR / f"{img['key']}.jpg").write_bytes(img["data"])
        (OUTPUT_DIR / f"{img['key']}.txt").write_text(caption)
        print(f"\n[{img['key']}]\n{caption}\n")

    print(f"{'=' * 60}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
