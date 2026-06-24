"""Strategy F on a broader sample of chinesememe rows (not just first 10)
to confirm it generalizes across the dataset."""

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
)


def call_qwen(messages, max_tokens=800):
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
    return r.json()["choices"][0]["message"]["content"]


def main():
    # Sample from multiple parts of the dataset to test generalization
    p = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/chinesememe")
    files = sorted(p.glob('train-*.parquet'))
    print(f"shards: {len(files)}")
    # Pick rows from spread-out indices
    SAMPLE_INDICES = [100, 250, 400, 600, 800, 1000, 1200, 1400]

    # Load whole shard 0 (it has 1517 rows per earlier check)
    tbl = pq.read_table(str(files[0]))
    total = tbl.num_rows
    rows = tbl.to_pydict()

    n_safe = 0
    n_nsfw = 0
    for idx in SAMPLE_INDICES:
        if idx >= total:
            continue
        img_bytes = rows["images"][idx][0]["bytes"]
        img_b64 = base64.b64encode(img_bytes).decode()
        texts = rows["texts"][idx]
        q = texts[0]["user"].strip()
        cn = texts[0]["assistant"].strip()

        print("=" * 90)
        print(f"ROW {idx} | English question: {q!r}")
        print(f"Original Chinese answer ({len(cn)} chars):")
        print(f"  {cn[:200]}...")
        print()
        try:
            out = call_qwen([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": q},
                ]},
            ])
            print(out[:1500])
            if "SAFETY: NSFW" in out:
                n_nsfw += 1
            elif "SAFETY: SAFE" in out:
                n_safe += 1
        except Exception as e:
            print(f"  ERR: {e}")
        print()

    print(f"=== summary: {n_safe} SAFE, {n_nsfw} NSFW (of {len(SAMPLE_INDICES)} sampled) ===")


if __name__ == "__main__":
    main()
