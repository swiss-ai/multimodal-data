"""Strategy G — Qwen generates a multi-turn conversation per meme image.

Each turn focuses on a different aspect:
  Turn 1: visual description (what's in the image)
  Turn 2: text extraction + translation
  Turn 3: cultural context / humor explanation
  Turn 4 (optional): usage / when would you send this meme

Output format: numbered turn blocks, easy to parse.
"""

import base64
import requests
from pathlib import Path

import pyarrow.parquet as pq


ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"

SYSTEM_PROMPT = (
    "You are an expert at explaining Chinese internet memes to non-Chinese speakers. "
    "For every meme image, follow this protocol exactly.\n\n"
    "STEP 1 — Safety check. Output one of:\n"
    "  **SAFETY: SAFE**\n"
    "  **SAFETY: NSFW**\n\n"
    "Mark NSFW only if the meme contains: sexual / nudity / sexualized minors, graphic "
    "violence or gore, self-harm or suicide content, hate speech or slurs, illegal drug "
    "glorification, or real-person harassment. Stylized profanity puns and culturally-edgy "
    "humor are SAFE.\n\n"
    "STEP 2 — If NSFW, write a one-sentence reason and STOP.\n\n"
    "STEP 3 — If SAFE, generate a multi-turn English conversation about this meme. "
    "Use this EXACT format (3 to 4 turn pairs):\n\n"
    "**TURN 1**\n"
    "USER: <natural English question asking the user to describe the visual>\n"
    "ASSISTANT: <concise English description of what is in the image>\n\n"
    "**TURN 2**\n"
    "USER: <natural English question asking about any Chinese text in the image>\n"
    "ASSISTANT: <OCR'd Chinese text verbatim + faithful English translation. "
    "If no Chinese text, say so.>\n\n"
    "**TURN 3**\n"
    "USER: <natural English question asking why the meme is funny or what cultural reference it uses>\n"
    "ASSISTANT: <explanation of wordplay, slang, cultural context, and humor — assume "
    "reader has no Chinese language or culture knowledge>\n\n"
    "**TURN 4** (optional, include only if it adds real value)\n"
    "USER: <natural English question asking when / in what situation someone would use this meme>\n"
    "ASSISTANT: <typical usage context — what conversation/emotion it expresses>\n\n"
    "Rules:\n"
    "  - Each USER question must be NATURAL English (varied phrasing, not template-like).\n"
    "  - Each ASSISTANT answer must be self-contained (don't reference 'as I said earlier').\n"
    "  - Keep each answer tight (2-4 sentences typical; longer only if needed)."
)


def call_qwen(messages, max_tokens=1500):
    r = requests.post(
        f"{ROUTER}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3,  # slight bump for varied phrasing
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    p = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/chinesememe")
    tbl = pq.read_table(str(sorted(p.glob('train-*.parquet'))[0]))
    rows = tbl.to_pydict()

    # Same 5 spread-out indices as the broader test
    INDICES = [100, 250, 400, 600, 800]

    for idx in INDICES:
        img_bytes = rows["images"][idx][0]["bytes"]
        img_b64 = base64.b64encode(img_bytes).decode()
        texts = rows["texts"][idx]
        original_q = texts[0]["user"].strip()

        print("=" * 90)
        print(f"ROW {idx} | original English Q: {original_q!r}")
        print()
        try:
            out = call_qwen([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": (
                        "Generate a multi-turn English conversation about this Chinese meme "
                        "following the protocol in the system prompt."
                    )},
                ]},
            ])
            print(out)
        except Exception as e:
            print(f"  ERR: {e}")
        print()


if __name__ == "__main__":
    main()
