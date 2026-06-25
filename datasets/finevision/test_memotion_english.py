"""Test Qwen3.6-27B's ability to explain English memes.

Uses the chinesememe Strategy H prompt template, adapted for English memes
(no Chinese-specific OCR language). Tests on 5 random memotion images.
"""

import base64
import random
import requests
from pathlib import Path

import pyarrow.parquet as pq


ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"

SYSTEM_PROMPT = (
    "You explain internet memes to readers who may not be familiar with the "
    "cultural references. For every meme image, follow this exact protocol.\n\n"
    "STEP 1 — Safety check. Output one of:\n"
    "  **SAFETY: SAFE**\n"
    "  **SAFETY: NSFW**\n\n"
    "Mark NSFW only for: sexual content / nudity / sexualized minors, graphic "
    "violence or gore, self-harm or suicide content, hate speech or ethnic slurs, "
    "illegal drug glorification, real-person harassment. Edgy humor and stylized "
    "profanity are SAFE.\n\n"
    "STEP 2 — If NSFW, write one-sentence reason and STOP.\n\n"
    "STEP 3 — If SAFE, generate a 2-turn English conversation in this EXACT format.\n\n"
    "**TURN 1**\n"
    "USER: <the exact user question you were given — do not rephrase>\n"
    "ASSISTANT: <ONE comprehensive paragraph (80-150 words) that includes:\n"
    "  - what is shown in the image (subjects, action, style),\n"
    "  - any text on the image, quoted verbatim,\n"
    "  - what the meme conveys / why it's funny / cultural reference if any.>\n\n"
    "**TURN 2**\n"
    "USER: <pick ONE follow-up angle most interesting for THIS specific meme; "
    "vary your choice across memes. Choose from (or invent similar):\n"
    "  - usage / situation: when would someone post this?\n"
    "  - cultural reference: what older meme / movie / character is this referencing?\n"
    "  - emotional read: what feeling does sharing this express?\n"
    "  - era / origin: is this a recent meme or older?\n"
    "  - audience fit: who's the typical audience?\n"
    "  - visual mechanic: why this specific template (panel layout, character, etc.)?\n"
    "  - non-obvious meaning: what's a deeper / ironic / second-layer reading?>\n"
    "ASSISTANT: <natural extension, 60-120 words, deepening the chosen angle>\n\n"
    "Rules:\n"
    "  - Turn 1 USER must be the exact original question — do not edit it.\n"
    "  - Turn 2 USER must feel like a natural human follow-up.\n"
    "  - Both ASSISTANT answers stand alone — no 'as I mentioned above'."
)

# Varied "explain this meme" questions to replace memotion's narrow sentiment prompts
QUESTION_POOL = [
    "What does this meme convey?",
    "Explain what makes this image funny or meaningful.",
    "Describe what you see in this meme image.",
    "What is happening in this meme image?",
    "Explain the content and meaning of this meme.",
    "What is the emotional message of this meme?",
    "What mood or sentiment does this meme capture?",
    "How does this meme work / what's the joke?",
    "What kind of reaction is this meme meant to provoke?",
    "Describe the characters, text, and context in this meme.",
]


def call_qwen(messages, max_tokens=1500):
    r = requests.post(
        f"{ROUTER}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    p = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/memotion")
    tbl = pq.read_table(str(sorted(p.glob('train-*.parquet'))[0]))
    print(f"memotion shard 0: {tbl.num_rows:,} rows")
    rows = tbl.to_pydict()

    # Sample varied indices across the shard
    rng = random.Random(42)
    indices = rng.sample(range(tbl.num_rows), 5)
    print(f"sampling rows: {indices}\n")

    for i, idx in enumerate(indices):
        img_bytes = rows["images"][idx][0]["bytes"]
        img_b64 = base64.b64encode(img_bytes).decode()
        original_q = rows["texts"][idx][0]["user"].strip()
        original_a = rows["texts"][idx][0]["assistant"].strip()

        # Replace narrow sentiment question with broader meme-explanation prompt
        new_q = QUESTION_POOL[i % len(QUESTION_POOL)]

        print("=" * 90)
        print(f"ROW {idx}")
        print(f"  ORIGINAL Q: {original_q[:200]!r}")
        print(f"  ORIGINAL A: {original_a[:200]!r}")
        print(f"  NEW Q:      {new_q!r}")
        print()
        try:
            out = call_qwen([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": (
                        f"Original user question for this meme: \"{new_q}\"\n\n"
                        "Generate the 2-turn conversation per the system protocol."
                    )},
                ]},
            ])
            print(out[:2200])
        except Exception as e:
            print(f"  ERR: {e}")
        print()


if __name__ == "__main__":
    main()
