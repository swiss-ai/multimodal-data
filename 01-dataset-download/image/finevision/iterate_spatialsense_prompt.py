"""Prompt-iteration harness for SpatialSense redistill.

Usage:
    # set ROUTER first (see /tmp/qwen_router.url once cluster is up)
    export QWEN_ROUTER="http://nidXXXXXX:8080"

    # Pick the prompt revision from PROMPTS dict, default = "v1"
    python iterate_spatialsense_prompt.py --prompt v1 --n 8

    # To compare two prompt versions side-by-side on the same rows:
    python iterate_spatialsense_prompt.py --prompt v1,v2 --n 6 --seed 42

The script:
  - Picks N random SpatialSense rows (deterministic via --seed)
  - For each row, sends (image, question, gold_label) to Qwen3.6-27B with the
    chosen system prompt
  - Prints side-by-side: original FineVision answer vs. Qwen redistill
  - Lets you read the outputs, tweak the prompt in PROMPTS below, rerun
"""

from __future__ import annotations
import argparse
import base64
import glob
import io
import json
import os
import random
import sys
from textwrap import indent
import pyarrow.parquet as pq
import requests

ROUTER = os.environ.get("QWEN_ROUTER", "").rstrip("/")
MODEL = "Qwen/Qwen3.6-27B-xyixuan"

SRC_DIR = "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/spatialsense"


# ── prompt revisions ────────────────────────────────────────────────────────
PROMPTS: dict[str, str] = {
    "v1": """You are a precise visual scene describer. Given an image and a Yes/No spatial-relation question, output a SPECIFIC grounded answer.

HARD RULES:
1. The first word of your answer MUST be exactly "{label}." (Yes. or No., capitalized, with a period).
2. NEVER use vague phrases like: "elsewhere", "different relationship", "different areas", "appears to be positioned elsewhere", "a different position". These are FORBIDDEN.
3. For each named object in the question, state its ACTUAL location in the image using concrete spatial terms: above, below, left of, right of, behind, in front of, on top of, under, beside, next to, inside, outside.
4. If "No", you MUST state what the ACTUAL relation between the two objects IS (not just that the asked relation is wrong).
5. Length: 1-3 sentences. Natural English.

EXAMPLES (good):
  Q: Is the lizard behind the tree?  Gold: No.
  A: No. The lizard is in front of the tree, sitting on the dirt to the left of the tree's base.

  Q: Is the cat on the ground?  Gold: Yes.
  A: Yes. The cat is lying on the ground, with its body resting on a brown tile floor next to a mirror.

EXAMPLES (bad — DO NOT produce these):
  A: No. The lizard is not behind the tree. From this viewpoint, the lizard appears to be positioned elsewhere in the scene.
  A: No. The spatial positioning shows a different relationship between these objects.

Now answer:
Question: {question}
Gold label: {label}.""",

    # v2 — tighten the forbidden list; ban hedging + axis-only patterns; anti-negation.
    "v2": """You are a precise visual scene describer. Output a SPECIFIC grounded spatial answer.

Output exactly 1-2 sentences. The first word MUST be "{label}." (capitalized, period).

FORBIDDEN PHRASES (your answer must contain NONE of these):
  - "elsewhere", "in a different area", "in a different position"
  - "different relationship", "different horizontal/vertical relationship"
  - "different arrangement", "different orientation", "different perspective"
  - "appears to be", "seems to be", "might be", "possibly", "perhaps", "may be"
  - "off to the side", "in another part of", "in a separate"
  - any sentence whose pattern is "X is not <relation> Y" (use the ACTUAL relation instead)

REQUIRED:
  - Use one concrete spatial term from this list to state the ACTUAL relation between the two named objects: "above", "below", "to the left of", "to the right of", "behind", "in front of", "on top of", "under", "beside", "next to", "inside", "outside", "between", "near".
  - For "No" labels: DO NOT include the asked relation in your sentence. State directly what the ACTUAL relation IS.

Correct examples:
  Q: Is the lizard behind the tree?  Gold: No.
  A: No. The lizard is in front of the tree, on the dirt to the left of its base.

  Q: Is the cat on the ground?  Gold: Yes.
  A: Yes. The cat is lying on the ground, with its body resting on a brown tile floor near a mirror.

  Q: Is the book on the shelf?  Gold: Yes.
  A: Yes. The book is on top of the wooden shelf, between two other books and below a vase.

  Q: Is the stone behind the lizard?  Gold: No.
  A: No. The stone is in front of the lizard, partially hidden by a fallen leaf.

INCORRECT (never produce):
  ✗ "No. The lizard is not behind the tree. From this viewpoint, the lizard appears to be positioned elsewhere in the scene."
  ✗ "No. The stone is not behind the lizard. The spatial positioning shows a different relationship."
  ✗ "No. They may be near each other, but the X is not on top of the Y."

Now answer:
Question: {question}
Gold label: {label}.""",

    # v3 — explicit structural slot + 6 few-shots; tightest format.
    "v3": """You're labeling spatial relations for a training dataset. Output ONE sentence in this exact form:

    <Label>. The <subject> is <spatial-phrase> the <reference-object>.

where:
  <Label> = "Yes" or "No" — given to you as the gold ground truth.
  <spatial-phrase> = one of {{above, below, to the left of, to the right of, behind, in front of, on top of, under, beside, next to, inside, outside, near}}, OR "between [obj1] and [obj2]".

You MAY add at most ONE additional short sentence that further grounds the position in the image (e.g. surface, neighboring objects). No more.

NEVER use: "elsewhere", "different relationship", "appears to", "seems to", "may be", "possibly", "off to the side", "in another part of", "perhaps", "in a different".

For "No" labels: NEVER say "X is not [asked-relation] Y". State the ACTUAL relation between the two objects directly.

Few-shot examples:
  Q: Is the lizard behind the tree?  Gold: No.
  A: No. The lizard is in front of the tree, on the dirt by the tree's base.

  Q: Is the cat on the ground?  Gold: Yes.
  A: Yes. The cat is lying on the ground on a brown tile floor.

  Q: Is the book on the shelf?  Gold: Yes.
  A: Yes. The book is on top of the wooden shelf, between two other books.

  Q: Is the stone behind the lizard?  Gold: No.
  A: No. The stone is in front of the lizard, partially hidden by a fallen leaf.

  Q: Is the boy on the gate?  Gold: Yes.
  A: Yes. The boy is on top of the gate, sitting astride it.

  Q: Is the tomato above the plate?  Gold: No.
  A: No. The tomato is to the right of the plate, on a wooden cutting board.

Now answer:
Q: {question}
Gold: {label}.""",
}


# ── helpers ─────────────────────────────────────────────────────────────────
def load_rows(n: int, seed: int) -> list[dict]:
    fs = sorted(glob.glob(f"{SRC_DIR}/train-*.parquet"))
    rows: list[dict] = []
    for f in fs:
        tbl = pq.read_table(f)
        for r in tbl.to_pylist():
            rows.append(r)
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


def to_data_url(image_dict: dict) -> str:
    b = image_dict["bytes"]
    return f"data:image/jpeg;base64,{base64.b64encode(b).decode()}"


def extract_label(answer_text: str) -> str:
    head = answer_text.strip()[:4].lower()
    if head.startswith("yes"):
        return "Yes"
    if head.startswith("no"):
        return "No"
    return "?"


def call_qwen(system_prompt: str, image_data_url: str, question: str, label: str,
              timeout: int = 60) -> str:
    if not ROUTER:
        return "[ROUTER not set; export QWEN_ROUTER=...]"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt.format(label=label, question=question)},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": f"{question}\nGold label: {label}."},
            ]},
        ],
        "max_tokens": 200,
        "temperature": 0.2,
        # Qwen3.6 defaults to thinking-mode; disable so answer lands in `content`.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = requests.post(f"{ROUTER}/v1/chat/completions", json=payload, timeout=timeout)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        c = msg.get("content") or msg.get("reasoning") or ""
        return c.strip()
    except Exception as e:
        return f"[error: {e}]"


# ── main loop ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1", help="comma-separated prompt revision keys to compare")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    versions = [v.strip() for v in args.prompt.split(",")]
    rows = load_rows(args.n, args.seed)

    for i, row in enumerate(rows):
        turns = row.get("texts") or []
        # Take just turn 0 for iteration
        turn0 = turns[0]
        q = (turn0.get("user") or "").replace("<image>\n", "").strip()
        a_original = (turn0.get("assistant") or "").strip()
        label = extract_label(a_original)
        if label == "?":
            continue
        img = row["images"][0]
        data_url = to_data_url(img)

        print("=" * 80)
        print(f"[{i+1}/{len(rows)}] gold_label={label}")
        print(f"  Q: {q}")
        print(f"  ORIGINAL (FineVision): {a_original}")
        for v in versions:
            sp = PROMPTS.get(v, "")
            out = call_qwen(sp, data_url, q, label) if sp.strip() else "[empty prompt]"
            print(f"\n  --- prompt {v!r} ---")
            print(indent(out, "    "))
        print()


if __name__ == "__main__":
    main()
