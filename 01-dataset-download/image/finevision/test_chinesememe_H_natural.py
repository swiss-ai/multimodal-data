"""Strategy H — 2-turn natural conversation flow.

- Turn 1 USER = original English question from the row (free diversity)
- Turn 1 ASSISTANT = comprehensive paragraph with:
    * visual description
    * if Chinese text: explicit "The text reads '<Chinese>', which means '<English>'"
    * brief humor / meaning
- Turn 2 USER = Qwen picks ONE varied follow-up angle, phrases naturally
- Turn 2 ASSISTANT = natural deepening (usage, cultural context, Western parallel, etc.)

The Turn 1 OCR+translation pattern teaches the student model OCR, translation,
visual description, AND humor explanation in one rich answer.
"""

import base64
import requests
from pathlib import Path

import pyarrow.parquet as pq


ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"

SYSTEM_PROMPT = (
    "You explain Chinese internet memes to non-Chinese English speakers. "
    "For every meme image, follow this exact protocol.\n\n"
    "STEP 1 — Safety check. Output one of:\n"
    "  **SAFETY: SAFE**\n"
    "  **SAFETY: NSFW**\n\n"
    "Mark NSFW only for: sexual content / nudity / sexualized minors, graphic violence "
    "or gore, self-harm or suicide content, hate speech or slurs, illegal drug glorification, "
    "real-person harassment. Edgy humor and stylized profanity puns are SAFE.\n\n"
    "STEP 2 — If NSFW, write one-sentence reason and STOP.\n\n"
    "STEP 3 — If SAFE, generate a 2-turn English conversation in this EXACT format.\n\n"
    "**TURN 1**\n"
    "USER: <the exact user question you were given — do not rephrase>\n"
    "ASSISTANT: <ONE comprehensive paragraph (80-150 words) that always includes:\n"
    "  - what is shown in the image (subjects, action, style),\n"
    "  - if the image has Chinese characters, quote them verbatim using this exact pattern: "
    "'The text on the image reads \"<verbatim Chinese>\", which means \"<faithful English translation>\".' "
    "If no Chinese text, say so naturally and skip this step.\n"
    "  - what the meme conveys / why it's funny (brief).>\n\n"
    "**TURN 2**\n"
    "USER: <pick ONE follow-up angle that is most interesting for THIS specific meme, "
    "and phrase it naturally as if a curious user replying. Vary your choice across memes — "
    "don't always pick the same angle. Choose from (or invent something similar to) one of:\n"
    "  - usage / situation: when would someone actually send this?\n"
    "  - Western parallel: is there an English-language equivalent?\n"
    "  - cultural depth: what's the deeper reference / history?\n"
    "  - emotional read: what feeling does sending this express?\n"
    "  - era / origin: is this a recent meme or older?\n"
    "  - audience fit: would it land with a Chinese friend / coworker?\n"
    "  - visual mechanic: why this specific template (panda head / cat / Ultraman / etc.)?\n"
    "  - slang breakdown: walk me through the wordplay token-by-token>\n"
    "ASSISTANT: <natural extension, 60-120 words, deepening the chosen angle>\n\n"
    "Rules:\n"
    "  - Turn 1 USER must be the exact original question — do not edit it.\n"
    "  - Turn 2 USER must feel like a natural human follow-up, not a templated prompt.\n"
    "  - Both ASSISTANT answers stand alone — no 'as I mentioned above'.\n"
    "  - Stay in English throughout (apart from the quoted Chinese characters)."
)


def call_qwen(messages, max_tokens=1500):
    r = requests.post(
        f"{ROUTER}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.4,  # bump for natural follow-up phrasing variety
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

    # Mix of indices from our earlier inspections (covers diverse meme types)
    INDICES = [100, 250, 400, 600, 800, 1000, 1200, 1400]

    for idx in INDICES:
        img_bytes = rows["images"][idx][0]["bytes"]
        img_b64 = base64.b64encode(img_bytes).decode()
        texts = rows["texts"][idx]
        original_q = texts[0]["user"].strip()

        print("=" * 90)
        print(f"ROW {idx} | original Q: {original_q!r}")
        print()
        try:
            out = call_qwen([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": (
                        f"Original user question for this meme: \"{original_q}\"\n\n"
                        "Generate the 2-turn conversation per the system protocol."
                    )},
                ]},
            ])
            print(out)
        except Exception as e:
            print(f"  ERR: {e}")
        print()


if __name__ == "__main__":
    main()
