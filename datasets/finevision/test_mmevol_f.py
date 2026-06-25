"""Strategy F adapted for mmevol — visual reasoning + grounding.

mmevol differs from chinesememe in three key ways:
  1. The original is GPT-4V multi-turn evolution prompts (we take turn 0)
  2. Assistant outputs contain bbox grounding (currently 0-1 floats)
  3. Wants reasoning chain, not just description

System prompt asks Qwen3.6-27B to:
  - Safety-gate (same protocol as chinesememe)
  - Generate reasoning chain
  - Use BLIP-style grounding: <object>name</object><bbox>[X1,Y1,X2,Y2]</bbox> with 0-1000 ints
  - Give final answer
"""

import base64
import requests
from pathlib import Path

import pyarrow.parquet as pq


ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"

SYSTEM_PROMPT = (
    "You are an expert at visual reasoning grounded in images. "
    "For every user question about an image, follow this protocol exactly.\n\n"
    "STEP 1 — Safety check. Output one of:\n"
    "  **SAFETY: SAFE**\n"
    "  **SAFETY: NSFW**\n\n"
    "Mark NSFW only if the image contains: sexual / nudity / sexualized minors, graphic violence "
    "or gore, self-harm or suicide content, hate speech or slurs, illegal drug glorification, or "
    "real-person harassment. Edgy humor, profanity, and culturally-edgy memes are SAFE.\n\n"
    "STEP 2 — If NSFW, write a one-sentence reason and STOP.\n\n"
    "STEP 3 — If SAFE, answer with this exact format:\n\n"
    "**Reasoning:**\n"
    "<concise step-by-step reasoning. When you mention any object, ground it with a "
    "bounding-box reference in this exact format: "
    "`<object>NAME</object><bbox>[X1,Y1,X2,Y2]</bbox>` where the four numbers are "
    "integers in the range 0-1000 (normalized image coordinates; X1,Y1 = top-left, "
    "X2,Y2 = bottom-right). Use real coordinates from the image — do not invent.>\n\n"
    "**Answer:**\n"
    "<final answer to the user's question — concise>"
)


def call_qwen(messages, max_tokens=1200):
    r = requests.post(
        f"{ROUTER}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]


def main():
    p = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/mmevol")
    tbl = pq.read_table(str(sorted(p.glob('train-*.parquet'))[0]))
    rows = tbl.slice(0, 5).to_pydict()

    for i in range(5):
        img_bytes = rows["images"][i][0]["bytes"]
        img_b64 = base64.b64encode(img_bytes).decode()
        texts = rows["texts"][i]
        # Take turn 0 as canonical question (the evolution prompts ask same content N ways)
        q0 = texts[0]["user"].strip()
        a0_orig = texts[0]["assistant"].strip()

        print("=" * 90)
        print(f"ROW {i} | turns: {len(texts)}")
        print(f"USER (turn 0): {q0[:250]!r}")
        print(f"ORIGINAL GPT-4V ANSWER ({len(a0_orig)} chars):")
        print(f"  {a0_orig[:400]}...")
        print()
        print("--- Qwen3.6-27B with system prompt ---")
        try:
            out = call_qwen([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": q0},
                ]},
            ])
            print(out[:2000])
        except Exception as e:
            print(f"  ERR: {e}")
        print()


if __name__ == "__main__":
    main()
