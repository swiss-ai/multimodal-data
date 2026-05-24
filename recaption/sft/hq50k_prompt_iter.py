#!/usr/bin/env python3
"""HQ-50K caption prompt iteration harness.

Loads a fixed set of diverse images once, runs every defined prompt version
on all of them in a single model session, and writes results to
  artifacts/hq50k_prompt_iter/{version}/

Usage:
    SFT_RECAPTION_ENABLE_THINKING=1 CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python scripts/hq50k_prompt_iter.py [--versions v1 v2 ...]

Omit --versions to run all defined prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DATASET_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/hf___YangQiee___HQ-50K/downloaded")
OUTPUT_ROOT = _ROOT / "artifacts" / "hq50k_prompt_iter"
MODEL_PATH = "/tmp/models/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
# Fixed image positions (shard_index, member_index_within_shard)
# Chosen to span shards and different positions for visual diversity
FIXED_IMAGE_POSITIONS = [
    (0, 300),
    (0, 1800),
    (0, 3500),
    (1, 400),
    (1, 2800),
    (2, 600),
    (2, 4000),
    (3, 1200),
    (4, 500),
    (4, 5000),
]

# ------------------------------------------------------------------ #
# Prompt versions — each iteration refines the previous
# ------------------------------------------------------------------ #

SCHEMA = '{"caption": "..."}'

PROMPTS: dict[str, str] = {
    "v1": f"""\
Write a detailed, accurate caption for this image.

The caption should:
- Identify the primary subject(s) with specific detail (species, breed, material, color)
- Describe the setting and environment
- Note any action, interaction, or event occurring
- Capture lighting, mood, or atmosphere where present
- Use precise, concrete language — never vague terms

Requirements:
- 3–5 sentences, continuous prose
- Do not open with "The image shows", "This is a", or "In this image"
- Do not reference the medium (photo, image, picture)
- Do not describe anything not clearly visible
- No markdown, no lists

Return exactly one JSON object: {SCHEMA}""",
    "v2": f"""\
You are captioning photographs for a prestigious nature and science publication.
Write the caption a skilled human editor would attach to this image.

The caption must:
- Open with the most visually striking or significant element, named precisely
- Ground the scene: environment, setting, time of day or season if discernible
- Describe any motion, behavior, or event unfolding in the frame
- Convey the visual atmosphere: light quality, depth, color, texture
- Close with a detail that rewards careful looking

Requirements:
- 3–5 sentences, vivid and specific prose
- Forbidden openers: "The image", "This photo", "In this image", "This is a", "Shown here"
- No reference to the image format, resolution, or medium
- Only describe what is unambiguously visible — no guesses about off-frame context
- No markdown, no bullet points in output

Return exactly one JSON object: {SCHEMA}""",
    "v3": f"""\
Write a precise, richly detailed caption for this image suitable for training a frontier \
vision-language model. The caption must read like polished human writing.

Cover each dimension in your caption:
• PRIMARY SUBJECT: exact identity and what it is doing (species not "animal"; \
"red-brick Victorian townhouse" not "building")
• SETTING: specific environment type, background, foreground elements, spatial depth
• ATMOSPHERE: light quality, time of day or season if visible, dominant colors, mood
• NOTABLE DETAIL: one specific texture, expression, scale cue, or compositional element \
that a lesser description would miss

Synthesise these into flowing literary prose — NOT a list in the output.

Hard rules:
- 4–6 sentences, unbroken paragraph
- Begin with the subject itself — not a meta-sentence about what the image depicts
- Precise vocabulary: "cascades" not "falls"; "burnt sienna" not "orange-ish"; \
"juvenile bald eagle" not "bird"
- Forbidden words in output: "various", "several", "some", "things", "can be seen", \
"is visible", "depicts", "shows"
- Never invent detail not unambiguously visible (location names, person names, dates)
- No markdown

Return exactly one JSON object: {SCHEMA}""",
    "v4": f"""\
You are producing the highest-quality caption annotations for a vision-language model's \
final training stage. These captions set the quality ceiling the model will learn from.

WHAT EXCELLENT LOOKS LIKE:
"A Siberian tiger crouches at the edge of a snow-dusted pine forest at dusk, its breath \
misting in the cold air as amber light catches the vivid orange of its flanks. The dense \
boreal canopy behind it recedes into blue shadow, punctuated by the pale trunks of birch \
trees. Snow has settled unevenly on the tiger's broad shoulders, and its amber eyes are \
fixed on something beyond the frame."

WHAT TO AVOID (with fixes):
✗ "A tiger in a forest" → ✓ "A Siberian tiger at the forest edge"
✗ "The image shows a dog running" → ✓ "A border collie sprints across..."
✗ "Various colorful flowers" → ✓ "Clusters of violet lavender and golden sunflowers"
✗ "It is a beautiful scene" → ✓ describe what makes it beautiful, specifically
✗ Invented details ("in Siberia", "in winter") → only what is visible

YOUR CAPTION:
- 4–6 sentences of precise, active, literary prose
- Open with the primary subject named with full specificity
- Include: subject + action, environment, light/atmosphere, one telling detail
- No forbidden openers: "The image", "This photo", "In this", "Here we see", "A photo of"
- No vague filler: "various", "several", "some", "beautiful", "stunning", "amazing"
- No medium references (photo, image, picture, frame)
- Describe only what is unambiguously visible

Return exactly one JSON object: {SCHEMA}""",
    "v5": f"""\
You are annotating images for a top-tier vision-language model's cooldown training phase. \
This is the highest-quality data the model will ever see. Write as a master photojournalist \
or natural history writer would caption this image for a Pulitzer-winning publication.

STRUCTURE OF AN OUTSTANDING CAPTION:
Sentence 1 — Subject: who or what, doing what, with what specific visual quality
Sentence 2 — Environment: where, what surrounds it, spatial relationships
Sentence 3 — Light and atmosphere: quality of light, time of day if visible, color palette, \
emotional tone
Sentence 4–5 — Depth: a compositional observation, a scale reference, a textural detail, \
or implied narrative that elevates description into meaning

ABSOLUTE RULES:
- 4–6 sentences, a single unbroken paragraph of prose
- NEVER open with: "The image", "This image", "This photo", "This photograph", "In this", \
"Here", "Shown", "A photo of", "A picture of", "An image of"
- NEVER use: "various", "several", "some", "things", "something", "can be seen", "is visible", \
"depicts", "shows", "illustrates", "beautiful", "stunning", "amazing", "incredible"
- NEVER reference the image medium, format, camera, or resolution
- NEVER invent: place names, personal names, dates, or any detail not visually confirmed
- Precision required: name species, breeds, architectural styles, materials, colors exactly
- Active verbs: "soars", "cascades", "huddles", "clings" — not "is", "are", "can be seen"

Return exactly one JSON object and nothing else: {SCHEMA}""",
}

# ------------------------------------------------------------------ #
# Image loading
# ------------------------------------------------------------------ #


def load_fixed_images() -> list[dict]:
    shards = sorted(DATASET_ROOT.glob("*.tar"))
    images: list[dict] = []
    for shard_idx, member_idx in FIXED_IMAGE_POSITIONS:
        if shard_idx >= len(shards):
            continue
        shard_path = shards[shard_idx]
        with tarfile.open(shard_path) as tf:
            jpg_members = [m for m in tf.getmembers() if m.name.endswith(".jpg")]
            if member_idx >= len(jpg_members):
                member_idx = len(jpg_members) - 1
            member = jpg_members[member_idx]
            data = tf.extractfile(member).read()
            images.append(
                {
                    "key": member.name.replace(".jpg", ""),
                    "shard": shard_path.name,
                    "data": data,
                }
            )
    return images


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #


def run_version(
    version: str,
    prompt: str,
    images: list[dict],
    engine,
    output_dir: Path,
) -> None:
    from sft_recaption.json_utils import extract_json_object
    from sft_recaption.runtime import to_data_url
    from sft_recaption.schemas import ImagePayload

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build conversations
    conversations = []
    for img in images:
        payload = ImagePayload(media_type="image/jpeg", data=img["data"])
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": to_data_url(payload)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

    outputs = engine.chat(
        conversations,
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )

    results = []
    for img, raw in zip(images, outputs):
        caption = None
        error = None
        try:
            parsed = extract_json_object(raw)
            caption = parsed.get("caption", "").strip()
        except Exception as exc:
            error = str(exc)

        result = {
            "version": version,
            "key": img["key"],
            "shard": img["shard"],
            "caption": caption,
            "error": error,
            "raw_output_chars": len(raw),
        }
        results.append(result)

        # Human-readable per-sample file
        sample_path = output_dir / f"{img['key']}.txt"
        with sample_path.open("w") as f:
            f.write(f"KEY:    {img['key']}  ({img['shard']})\n")
            f.write(f"STATUS: {'OK' if caption else 'ERROR: ' + str(error)}\n\n")
            f.write("CAPTION:\n")
            f.write(caption or "(none)")
            f.write("\n\n--- RAW MODEL OUTPUT ---\n")
            f.write(raw)

    # JSONL summary
    with (output_dir / "results.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print to stdout for live monitoring
    ok = sum(1 for r in results if r["caption"])
    print(f"\n{'=' * 60}")
    print(f"VERSION {version}  ({ok}/{len(results)} parsed)")
    print(f"Output: {output_dir}")
    print("=" * 60)
    for r in results:
        print(f"\n  [{r['key']}]")
        if r["caption"]:
            print(f"  {r['caption']}")
        else:
            print(f"  ERROR: {r['error']}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--versions",
        nargs="*",
        default=None,
        help="Prompt versions to run (default: all defined)",
    )
    args = parser.parse_args()

    versions_to_run = args.versions or list(PROMPTS.keys())
    unknown = [v for v in versions_to_run if v not in PROMPTS]
    if unknown:
        print(f"Unknown versions: {unknown}. Available: {list(PROMPTS)}")
        sys.exit(1)

    print(f"Loading {len(FIXED_IMAGE_POSITIONS)} fixed test images...")
    images = load_fixed_images()
    print(f"Loaded {len(images)} images from {DATASET_ROOT.name}")

    from sft_recaption.config import ModelConfig
    from sft_recaption.runtime import VLLMChatEngine, configure_worker_environment

    configure_worker_environment(0)

    # Enable thinking for higher quality
    engine = VLLMChatEngine(
        ModelConfig(
            model_repo=MODEL_PATH,
            tensor_parallel_size=1,
            max_num_seqs=4,
            enforce_eager=False,
            download_dir=Path("/tmp/models"),
            chat_template_kwargs={"enable_thinking": True},
        )
    )

    for version in versions_to_run:
        prompt = PROMPTS[version]
        output_dir = OUTPUT_ROOT / version
        print(f"\nRunning {version}...")
        run_version(version, prompt, images, engine, output_dir)

    print(f"\n\nAll done. Results under: {OUTPUT_ROOT}")
    print("Directories:")
    for v in versions_to_run:
        print(f"  {OUTPUT_ROOT / v}")


if __name__ == "__main__":
    main()
