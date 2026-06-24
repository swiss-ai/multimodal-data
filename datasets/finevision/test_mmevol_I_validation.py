"""Validate Strategy I (mmevol re-distill with BLIP grounding) on 5 rows.

Tests whether the few-shot example in the system prompt fixes the format
compliance issues we saw in earlier Strategy F:
  - <object>NAME</object>...<bbox>[X,Y,X,Y]</bbox> with proper brackets
  - 0-1000 integer coords (not 0-1 floats)
  - Skips bbox for abstract questions
"""

import base64
import requests
import re
from pathlib import Path

import pyarrow.parquet as pq


ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"

SYSTEM_PROMPT = (
    "You are an expert at visual reasoning grounded in images. "
    "For every user question about an image, follow this exact protocol.\n\n"
    "STEP 1 — Safety check. Output one of:\n"
    "  **SAFETY: SAFE**\n"
    "  **SAFETY: NSFW**\n\n"
    "Mark NSFW only for: sexual / nudity / sexualized minors, graphic violence or gore, "
    "self-harm or suicide content, hate speech or slurs, illegal drug glorification, "
    "real-person harassment. Edgy or stylized humor is SAFE.\n\n"
    "STEP 2 — If NSFW, write one-sentence reason and STOP.\n\n"
    "STEP 3 — If SAFE, answer with EXACTLY this format:\n\n"
    "**Reasoning:**\n"
    "<step-by-step reasoning. When you reference an object, use this EXACT inline format: "
    "<object>NAME</object><bbox>[X1, Y1, X2, Y2]</bbox> where NAME is a short noun phrase "
    "and X1,Y1,X2,Y2 are INTEGERS in 0-1000 (normalized image coords; top-left = 0,0).>\n\n"
    "**Answer:**\n"
    "<clear final answer to the user's question>\n\n"
    "EXAMPLE OUTPUT (study the bbox format carefully):\n\n"
    "**SAFETY: SAFE**\n\n"
    "**Reasoning:**\n"
    "The image shows a basketball court. I can see <object>man in suit</object>"
    "<bbox>[347, 64, 638, 976]</bbox> standing among several people in athletic "
    "wear, including <object>basketball player</object><bbox>[52, 272, 252, 764]</bbox>. "
    "The mismatch between his formal clothing and the casual sports setting is the "
    "key visual contrast.\n\n"
    "**Answer:**\n"
    "The man in the suit seems out of place because he is wearing formal business "
    "attire while everyone else is dressed for basketball.\n\n"
    "Rules:\n"
    "  - Bbox must use exact format: <object>NAME</object><bbox>[X,Y,X,Y]</bbox> with [ and ] brackets.\n"
    "  - Coords are 0-1000 integers, NOT 0-1 floats.\n"
    "  - Only ground objects you can actually see — do not invent coordinates.\n"
    "  - For abstract questions (e.g. 'why', 'explain reasons') with no visual object to ground, "
    "you may skip bbox tags and just reason in plain text."
)


def call_qwen(messages, max_tokens=1500):
    r = requests.post(
        f"{ROUTER}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# Format compliance regex (the "ideal" form we want)
IDEAL_BBOX_RE = re.compile(r"<object>([^<]+)</object><bbox>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</bbox>")
# Loose detection (any bbox-ish pattern)
ANY_BBOX_RE = re.compile(r"<bbox[^>]*>[^<]*</?bbox>", re.IGNORECASE)


def audit(text: str) -> dict:
    ideal = IDEAL_BBOX_RE.findall(text)
    any_bbox = ANY_BBOX_RE.findall(text)
    n_ideal = len(ideal)
    n_any = len(any_bbox)
    # Check coord range
    bad_range = 0
    for _, x1, y1, x2, y2 in ideal:
        coords = [int(x1), int(y1), int(x2), int(y2)]
        if any(c < 0 or c > 1000 for c in coords):
            bad_range += 1
    return {
        "ideal_format_count": n_ideal,
        "total_bbox_attempts": n_any,
        "compliance_rate": n_ideal / max(1, n_any),
        "out_of_range": bad_range,
    }


def main():
    p = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/mmevol")
    tbl = pq.read_table(str(sorted(p.glob('train-*.parquet'))[0]))
    rows = tbl.slice(0, 5).to_pydict()

    for i in range(5):
        img_bytes = rows["images"][i][0]["bytes"]
        img_b64 = base64.b64encode(img_bytes).decode()
        q = rows["texts"][i][0]["user"].strip()

        print("=" * 90)
        print(f"ROW {i} | Q: {q[:150]!r}")
        print()
        try:
            out = call_qwen([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": q},
                ]},
            ])
            stats = audit(out)
            print(f"--- AUDIT: ideal_format={stats['ideal_format_count']} / "
                  f"any_bbox={stats['total_bbox_attempts']} = "
                  f"{stats['compliance_rate']*100:.0f}% compliance, "
                  f"out_of_range={stats['out_of_range']} ---")
            print()
            print(out[:2000])
        except Exception as e:
            print(f"  ERR: {e}")
        print()


if __name__ == "__main__":
    main()
