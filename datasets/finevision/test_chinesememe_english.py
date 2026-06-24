"""Compare 3 prompt strategies for re-explaining Chinese memes in English.

A) Image + original English question, NO Chinese reference → fresh English
B) Image + English question + Chinese reference → translation/preservation
C) Image + English question + Chinese ref + structured request (visual + text + humor)
"""

import base64
import io
import json
import requests
from pathlib import Path

import pyarrow.parquet as pq

# Router URL is auto-detected by the smart poller and saved to /tmp/qwen_router.url
ROUTER = Path("/tmp/qwen_router.url").read_text().strip() if Path("/tmp/qwen_router.url").exists() else "http://172.28.44.16:30000"
MODEL = "Qwen/Qwen3.6-27B-smatrenok"

# Three prompt strategies
STRATEGIES = {
    "A_imageonly": lambda img_b64, q, cn: [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": (
                f"{q}\n\n"
                "Please respond in clear English suitable for someone who does not understand Chinese. "
                "Translate any Chinese text in the image and explain any cultural references."
            )},
        ]},
    ],
    "B_with_chinese_ref": lambda img_b64, q, cn: [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": (
                f"{q}\n\n"
                "For reference, here is an existing Chinese description of this meme:\n"
                f"\"{cn}\"\n\n"
                "Please provide a clear English explanation that a non-Chinese reader can understand. "
                "Faithfully translate any text in the image and explain the cultural context."
            )},
        ]},
    ],
    "C_structured": lambda img_b64, q, cn: [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": (
                f"{q}\n\n"
                "For reference, the existing Chinese description is:\n"
                f"\"{cn}\"\n\n"
                "Provide an English explanation with these three labeled sections:\n"
                "1. **Visual:** what is shown in the image\n"
                "2. **Text on image:** English translation of any Chinese text overlay\n"
                "3. **Cultural context & humor:** explain wordplay, references, and why it's funny — "
                "for someone unfamiliar with Chinese internet culture"
            )},
        ]},
    ],
    "D_extract_translate_explain": lambda img_b64, q, cn: [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": (
                f"{q}\n\n"
                "Read the meme image and answer in this exact format:\n\n"
                "**Chinese text (verbatim, OCR'd from image):**\n"
                "<the Chinese characters that appear on the image — if none, write \"(none)\">\n\n"
                "**English translation:**\n"
                "<faithful English translation of those Chinese characters>\n\n"
                "**Visual description (English):**\n"
                "<one paragraph describing what is in the image>\n\n"
                "**Cultural context & humor (English):**\n"
                "<one paragraph explaining wordplay, references, internet slang, and why it's funny "
                "— assume the reader has no knowledge of Chinese language or culture>"
            )},
        ]},
    ],
    "E_with_nsfw_gate": lambda img_b64, q, cn: [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": (
                f"{q}\n\n"
                "FIRST, check if the meme is safe-for-work training content. Output:\n\n"
                "**SAFETY: <SAFE | NSFW>**\n\n"
                "A meme is NSFW if it contains ANY of:\n"
                "  - sexual content, nudity, or sexualized minors\n"
                "  - graphic violence, gore, or self-harm/suicide jokes\n"
                "  - hate speech, racism, or ethnic slurs\n"
                "  - illegal drug glorification\n"
                "  - real-person harassment or non-consensual imagery\n"
                "If unsafe, write \"**SAFETY: NSFW**\" then a brief reason and STOP. Do not describe further.\n\n"
                "If SAFE, write \"**SAFETY: SAFE**\" then continue with this exact format:\n\n"
                "**Chinese text (verbatim, OCR'd from image):**\n"
                "<Chinese chars on the image — if none, write \"(none)\">\n\n"
                "**English translation:**\n"
                "<faithful English translation>\n\n"
                "**Visual description (English):**\n"
                "<one paragraph>\n\n"
                "**Cultural context & humor (English):**\n"
                "<one paragraph — assume reader has no Chinese knowledge>"
            )},
        ]},
    ],
    "F_system_prompt": lambda img_b64, q, cn: [
        {"role": "system", "content": (
            "You are an expert at explaining Chinese internet memes to non-Chinese speakers. "
            "For every meme image, follow this protocol exactly.\n\n"
            "STEP 1 — Safety check. Output one of:\n"
            "  **SAFETY: SAFE**\n"
            "  **SAFETY: NSFW**\n\n"
            "Mark NSFW if the meme contains ANY of: sexual content / nudity / sexualized minors, "
            "graphic violence or gore, self-harm or suicide jokes, hate speech or ethnic slurs, "
            "illegal drug glorification, real-person harassment or non-consensual imagery. "
            "Stylized profanity puns and culturally-edgy humor are SAFE — only flag genuine harm content.\n\n"
            "STEP 2 — If NSFW, write a one-sentence reason and STOP.\n\n"
            "STEP 3 — If SAFE, output exactly these four labeled sections (each one paragraph):\n\n"
            "**Chinese text (verbatim, OCR'd from image):**\n"
            "<Chinese characters on the image, verbatim — write \"(none)\" if no text>\n\n"
            "**English translation:**\n"
            "<faithful English translation of the OCR'd text>\n\n"
            "**Visual description (English):**\n"
            "<concise description of what is in the image>\n\n"
            "**Cultural context & humor (English):**\n"
            "<explain wordplay, slang, cultural references, and why the meme is funny — "
            "assume the reader has no knowledge of Chinese language or culture>"
        )},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": q},
        ]},
    ],
}


def call_qwen(messages, max_tokens=800):
    """Call Qwen with thinking DISABLED (we want direct caption output, not CoT)."""
    r = requests.post(
        f"{ROUTER}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
            # Disable Qwen3 reasoning to save tokens + get direct answer
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    if msg.get("reasoning_content"):
        return f"[thinking={len(msg['reasoning_content'])}c] {msg['content']}"
    return msg["content"]


def main():
    p = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/chinesememe")
    tbl = pq.read_table(str(sorted(p.glob('train-*.parquet'))[0]))
    rows = tbl.slice(0, 10).to_pydict()

    for i in range(10):
        img_bytes = rows["images"][i][0]["bytes"]
        img_b64 = base64.b64encode(img_bytes).decode()
        texts = rows["texts"][i]
        q = texts[0]["user"].strip()
        cn = texts[0]["assistant"].strip()

        print("=" * 90)
        print(f"ROW {i} | English question: {q!r}")
        print(f"Original Chinese answer ({len(cn)} chars):")
        print(f"  {cn[:300]}...")
        print()

        for name, build in [(k, v) for k, v in STRATEGIES.items() if k.startswith("F_")]:
            print(f"--- Strategy {name} ---")
            try:
                out = call_qwen(build(img_b64, q, cn))
                print(out[:1500])
            except Exception as e:
                print(f"  ERR: {e}")
            print()


if __name__ == "__main__":
    main()
