"""Re-distill memotion with Qwen3.6-27B (Strategy H, English meme explanations).

Drops memotion's narrow sentiment-classification labels (humor/sarcasm/sentiment
categoricals) and generates rich 2-turn English explanations of each meme:
visual + OCR + cultural reference + humor mechanism + natural follow-up.

NSFW criteria are TIGHTENED compared to chinesememe — adds ethnic/cultural
stereotyping as a filter target since English memes more often punch down on
nationalities/religions/cultures.

Run:
    python redistill_memotion_qwen.py            # full (6.8K rows, ~5-10 min)
    python redistill_memotion_qwen.py --shards 0  # sanity test
"""

from __future__ import annotations
import argparse
import base64
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"
IN_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/memotion")
OUT_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/memotion_recap_en")

# Varied "explain this meme" prompts to replace memotion's narrow sentiment ones.
# Distributed deterministically by row index (i % len) so coverage stays uniform.
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

SYSTEM_PROMPT = (
    "You explain internet memes to readers who may not be familiar with the "
    "cultural references. For every meme image, follow this exact protocol.\n\n"
    "STEP 1 — Safety check. Output one of:\n"
    "  **SAFETY: SAFE**\n"
    "  **SAFETY: NSFW**\n\n"
    "Mark NSFW if the meme contains ANY of:\n"
    "  - sexual content / nudity / sexualized minors\n"
    "  - graphic violence or gore\n"
    "  - self-harm or suicide jokes\n"
    "  - hate speech, ethnic slurs, racist content\n"
    "  - illegal drug glorification\n"
    "  - real-person harassment or non-consensual imagery\n"
    "  - ETHNIC, RACIAL, RELIGIOUS, OR NATIONAL STEREOTYPING used as humor "
    "(e.g., 'a person from country X always does Y', joking about cultural "
    "practices to demean them, punching-down humor at minority groups)\n"
    "  - cultural slurs or jokes that rely on prejudicial framing\n\n"
    "Edgy humor, self-deprecation, profanity puns, and absurdist memes are "
    "SAFE — only flag content that punches down on identity groups or treats "
    "real groups as caricatures for laughs.\n\n"
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

OUTPUT_SCHEMA = pa.schema([
    ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    ("texts", pa.list_(pa.struct([("user", pa.string()), ("assistant", pa.string())]))),
    ("source", pa.string()),
])


def call_qwen(img_b64: str, q: str, max_tokens: int = 1500, retries: int = 3) -> str | None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": (
                    f"Original user question for this meme: \"{q}\"\n\n"
                    "Generate the 2-turn conversation per the system protocol."
                )},
            ]},
        ],
        "temperature": 0.4,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(retries):
        try:
            r = requests.post(f"{ROUTER}/v1/chat/completions", json=payload, timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))


_TURN_RE = re.compile(
    r"\*\*TURN 1\*\*\s*\n"
    r"USER:\s*(.+?)\n"
    r"ASSISTANT:\s*(.+?)\n\s*"
    r"\*\*TURN 2\*\*\s*\n"
    r"USER:\s*(.+?)\n"
    r"ASSISTANT:\s*(.+?)\Z",
    re.DOTALL,
)


def parse(text: str):
    if not text:
        return "MALFORMED", None
    if "SAFETY: NSFW" in text:
        return "NSFW", None
    if "SAFETY: SAFE" not in text:
        return "MALFORMED", None
    m = _TURN_RE.search(text)
    if not m:
        return "MALFORMED", None
    return "SAFE", [
        {"user": m.group(1).strip(), "assistant": m.group(2).strip()},
        {"user": m.group(3).strip(), "assistant": m.group(4).strip()},
    ]


def process_shard(shard_path: Path, out_path: Path, concurrency: int) -> dict:
    if out_path.exists():
        return {"shard": shard_path.name, "skipped": True}
    tbl = pq.read_table(str(shard_path))
    rows = tbl.to_pylist()
    if not rows:
        return {"shard": shard_path.name, "empty": True}

    jobs = []
    for i, row in enumerate(rows):
        try:
            img_b64 = base64.b64encode(row["images"][0]["bytes"]).decode()
            # Replace narrow sentiment Q with varied meme-explanation prompt
            new_q = QUESTION_POOL[i % len(QUESTION_POOL)]
            jobs.append((img_b64, new_q, row))
        except Exception:
            continue

    out_rows = []
    n_safe = n_nsfw = n_malformed = n_err = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(call_qwen, b64, q): (b64, q, orig) for b64, q, orig in jobs}
        for fut in as_completed(futures):
            _, q, orig = futures[fut]
            try:
                raw = fut.result()
            except Exception:
                n_err += 1
                continue
            verdict, turns = parse(raw)
            if verdict == "SAFE":
                n_safe += 1
                out_rows.append({
                    "images": orig["images"],
                    "texts": turns,
                    "source": orig["source"],
                })
            elif verdict == "NSFW":
                n_nsfw += 1
            else:
                n_malformed += 1

    if out_rows:
        new_tbl = pa.Table.from_pylist(out_rows, schema=OUTPUT_SCHEMA)
        tmp = out_path.with_suffix(".parquet.tmp")
        pq.write_table(new_tbl, str(tmp), compression="zstd")
        tmp.rename(out_path)

    return {
        "shard": shard_path.name,
        "in": len(rows),
        "out": len(out_rows),
        "safe": n_safe,
        "nsfw": n_nsfw,
        "malformed": n_malformed,
        "errors": n_err,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--shards", type=str, default=None)
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(IN_ROOT.glob("train-*.parquet"))
    if args.shards:
        idxs = set(int(x) for x in args.shards.split(","))
        shard_files = [f for f in shard_files if int(re.search(r"train-(\d+)-of-", f.name).group(1)) in idxs]

    print(f"=== redistill memotion: {len(shard_files)} shards, concurrency={args.concurrency} ===\n", flush=True)
    t0 = time.time()
    total = {"in": 0, "out": 0, "safe": 0, "nsfw": 0, "malformed": 0, "errors": 0, "skipped": 0}

    for sf in shard_files:
        out_path = OUT_ROOT / sf.name
        s0 = time.time()
        stats = process_shard(sf, out_path, args.concurrency)
        dt = time.time() - s0
        if stats.get("skipped"):
            print(f"  [skip] {sf.name}", flush=True); total["skipped"] += 1; continue
        if stats.get("empty"):
            print(f"  [empty] {sf.name}", flush=True); continue
        for k in ("in", "out", "safe", "nsfw", "malformed", "errors"):
            total[k] += stats[k]
        kept = 100 * stats["out"] / max(1, stats["in"])
        print(f"  [{sf.name}] in={stats['in']:>5,} out={stats['out']:>5,} "
              f"({kept:>5.1f}% kept) safe={stats['safe']} nsfw={stats['nsfw']} "
              f"malformed={stats['malformed']} err={stats['errors']} in {dt:.0f}s",
              flush=True)

    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed:.0f}s ===")
    print(f"  total in:        {total['in']:,}")
    print(f"  total out:       {total['out']:,} ({100*total['out']/max(1,total['in']):.1f}% kept)")
    print(f"  safe:            {total['safe']:,}")
    print(f"  nsfw filtered:   {total['nsfw']:,}")
    print(f"  malformed:       {total['malformed']:,}")
    print(f"  request errors:  {total['errors']:,}")


if __name__ == "__main__":
    main()
