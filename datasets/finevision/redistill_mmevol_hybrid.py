"""Hybrid re-distill: regen ALL grounding rows; judge non-grounding rows
(keep on pass, DROP on fail — no quality-fix regeneration).

Pipeline per row:
  IF the original GPT-4V answer contains 0-1 float bbox coords:
      → REGEN with Qwen (Strategy I prompt) — needed for 0-1000 BLIP format
      anyway, no point judging first.
  ELSE (non-grounding row):
      → JUDGE: Qwen scores (factual, grounding, reasoning, format) 1-5 Likert
      → If all 3 content scores >= 3 → KEEP original GPT-4V answer
      → If any content score <= 2 → DROP the row (no regeneration)

Why drop instead of regen on quality failure: if the judge thinks the original
is unreliable, regenerating risks the same hallucination class with a different
teacher. Cleaner to drop the row entirely than to backfill with a different
guess. Grounding rows are special — they only fail format (not content), so
regen is safe + necessary.

Expected outcome (~137K input):
  ~12K regenerated (grounding format fix)
  ~100K kept original (passed judge)
  ~25K dropped (failed judge)
  ~112K kept overall (~82% retention)

Output: processed/sft/finevision/mmevol_judged/

Run:
    python redistill_mmevol_hybrid.py             # full
    python redistill_mmevol_hybrid.py --shards 0 --concurrency 50  # sanity
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
IN_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/mmevol")
OUT_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/mmevol_judged")

# Score threshold — keep rows where all 3 content dims are >= this; format is
# always ~5.0 so we don't gate on it.
PASS_THRESHOLD = 3
CONTENT_DIMS = ("factual", "grounding", "reasoning")

JUDGE_PROMPT = (
    "You are an expert evaluator of vision-language model training data. "
    "Given an image, a user question, and an assistant answer, score on FOUR "
    "dimensions (1-5 Likert, 5 = excellent, 1 = poor):\n"
    "  1. **Factual correctness**: Does the answer correctly address the question?\n"
    "  2. **Visual grounding**: Do object/location references match the image?\n"
    "  3. **Reasoning quality**: Is the explanation sound and useful?\n"
    "  4. **Format quality**: Is the output clean and readable?\n\n"
    "Output EXACTLY:\n"
    "**Factual:** <1-5>\n"
    "**Grounding:** <1-5>\n"
    "**Reasoning:** <1-5>\n"
    "**Format:** <1-5>"
)

REGEN_PROMPT = (
    "You are an expert at visual reasoning grounded in images. "
    "For every user question about an image, follow this exact protocol.\n\n"
    "STEP 1 — Safety check. Output one of:\n"
    "  **SAFETY: SAFE**\n"
    "  **SAFETY: NSFW**\n\n"
    "Mark NSFW only for: sexual / nudity / sexualized minors, graphic violence "
    "or gore, self-harm or suicide content, hate speech or slurs, illegal drug "
    "glorification, real-person harassment. Edgy or stylized humor is SAFE.\n\n"
    "STEP 2 — If NSFW, write one-sentence reason and STOP.\n\n"
    "STEP 3 — If SAFE, generate a 2-turn conversation in this EXACT format:\n\n"
    "**TURN 1**\n"
    "USER: <the exact user question you were given — do not rephrase>\n"
    "ASSISTANT:\n"
    "**Reasoning:**\n"
    "<step-by-step reasoning. When you reference an object, use this EXACT inline format: "
    "<object>NAME</object><bbox>[X1, Y1, X2, Y2]</bbox> where NAME is a short noun phrase "
    "and X1,Y1,X2,Y2 are INTEGERS in 0-1000 (normalized image coords; top-left = 0,0).>\n\n"
    "**Answer:**\n"
    "<clear final answer to the user's question>\n\n"
    "**TURN 2**\n"
    "USER: <pick ONE natural follow-up about the same image — vary your choice across rows. "
    "Examples (don't always use the same one):\n"
    "  - 'Where exactly is [object] located relative to other objects?' (spatial)\n"
    "  - 'What else is notable in the scene?' (broader observation)\n"
    "  - 'Could you describe what's happening around [object]?' (context)\n"
    "  - 'Is there an alternative interpretation?' (epistemic)\n"
    "  - 'What can you tell about the lighting / time of day / setting?' (atmospheric)>\n"
    "ASSISTANT:\n"
    "**Reasoning:**\n"
    "<reasoning for the follow-up, using same <object><bbox> grounding format where applicable.>\n\n"
    "**Answer:**\n"
    "<concise answer to the follow-up>\n\n"
    "EXAMPLE OUTPUT:\n\n"
    "**SAFETY: SAFE**\n\n"
    "**TURN 1**\n"
    "USER: Why does the man in a suit and tie seem out of place in this scenario?\n"
    "ASSISTANT:\n"
    "**Reasoning:**\n"
    "The image shows a basketball court. I can see <object>man in suit</object>"
    "<bbox>[347, 64, 638, 976]</bbox> standing among several people in athletic "
    "wear, including <object>basketball player</object><bbox>[52, 272, 252, 764]</bbox>. "
    "The mismatch between his formal clothing and the casual sports setting is the "
    "key visual contrast.\n\n"
    "**Answer:**\n"
    "The man in the suit seems out of place because he is wearing formal business "
    "attire while everyone else is dressed for basketball.\n\n"
    "**TURN 2**\n"
    "USER: Where exactly is the man in the suit positioned relative to the players?\n"
    "ASSISTANT:\n"
    "**Reasoning:**\n"
    "Looking at the spatial layout, <object>man in suit</object>"
    "<bbox>[347, 64, 638, 976]</bbox> stands on the right side of the court, "
    "facing the camera. The closest <object>basketball player</object>"
    "<bbox>[52, 272, 252, 764]</bbox> is to his left, mid-action.\n\n"
    "**Answer:**\n"
    "He stands on the right side of the court, with the nearest active player "
    "to his left.\n\n"
    "Rules:\n"
    "  - Turn 1 USER must be the EXACT original question — do not rephrase.\n"
    "  - Turn 2 USER must be a natural English follow-up; vary the angle across rows.\n"
    "  - Bbox format: <object>NAME</object><bbox>[X,Y,X,Y]</bbox> with [ and ] brackets.\n"
    "  - Coords are 0-1000 integers, NOT 0-1 floats.\n"
    "  - Only ground visible objects — do not invent coordinates.\n"
    "  - For abstract questions with no visual object to ground, skip bbox tags."
)

OUTPUT_SCHEMA = pa.schema([
    ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    ("texts", pa.list_(pa.struct([("user", pa.string()), ("assistant", pa.string())]))),
    ("source", pa.string()),
])


def post(payload, retries=3, timeout=120):
    for attempt in range(retries):
        try:
            r = requests.post(f"{ROUTER}/v1/chat/completions", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))


def judge(img_b64: str, q: str, a: str):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": f"USER QUESTION: {q}\n\nASSISTANT ANSWER:\n{a[:2500]}\n\nScore this answer."},
            ]},
        ],
        "temperature": 0.1,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return post(payload, timeout=90)


def regen(img_b64: str, q: str):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": REGEN_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": q},
            ]},
        ],
        "temperature": 0.3,
        "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return post(payload, timeout=180)


_SCORE_RE = {
    d: re.compile(rf"\*\*{d.capitalize()}:\*\*\s*(\d)")
    for d in ("factual", "grounding", "reasoning", "format")
}
# 2-turn parse for regen output
_TURN_RE = re.compile(
    r"\*\*TURN 1\*\*\s*\n"
    r"USER:\s*(.+?)\n"
    r"ASSISTANT:\s*\n?(.+?)\n\s*"
    r"\*\*TURN 2\*\*\s*\n"
    r"USER:\s*(.+?)\n"
    r"ASSISTANT:\s*\n?(.+?)\Z",
    re.DOTALL,
)


def parse_judge(text):
    if not text:
        return None
    scores = {}
    for d, rx in _SCORE_RE.items():
        m = rx.search(text)
        if not m:
            return None
        scores[d] = int(m.group(1))
    return scores


def parse_regen(text):
    """Parse 2-turn regen output. Returns (verdict, [{user, assistant}, ...]) or (status, None)."""
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


# Detect rows whose GPT-4V answer already contains 0-1 float bboxes — these
# always get regenerated (no point judging since format is wrong regardless).
_GROUNDING_PATTERN = re.compile(r"\[\s*0\.\d+\s*,\s*0\.\d+\s*,\s*0\.\d+\s*,\s*0\.\d+\s*\]")


def _do_regen(img_b64, q):
    verdict, regenerated = parse_regen(regen(img_b64, q))
    return verdict, regenerated


def process_one(img_b64, q, original_a):
    """Grounding rows → 2-turn regen. Non-grounding → judge turn 0 → keep|drop.

    Returns dict with `texts` field = list[{user, assistant}] when status implies
    output is written. For "kept", the caller substitutes the ORIGINAL row's
    full multi-turn texts (preserving mmevol's native multi-turn structure).
    """
    if _GROUNDING_PATTERN.search(original_a):
        verdict, turns = _do_regen(img_b64, q)
        if verdict == "SAFE":
            return {"status": "regenerated_grounding", "texts": turns}
        if verdict == "NSFW":
            return {"status": "nsfw_dropped", "texts": None}
        return {"status": "regen_malformed", "texts": None}

    # Non-grounding row → judge turn-0 as proxy for row quality
    scores = parse_judge(judge(img_b64, q, original_a))
    if scores is None:
        return {"status": "judge_failed", "texts": None}
    if all(scores[d] >= PASS_THRESHOLD for d in CONTENT_DIMS):
        return {"status": "kept", "texts": None, "scores": scores}  # caller uses original row's full texts
    return {"status": "dropped_low_quality", "texts": None, "scores": scores}


def process_shard(shard_path, out_path, concurrency):
    if out_path.exists():
        return {"shard": shard_path.name, "skipped": True}
    tbl = pq.read_table(str(shard_path))
    rows = tbl.to_pylist()
    if not rows:
        return {"shard": shard_path.name, "empty": True}

    jobs = []
    for row in rows:
        try:
            jobs.append((
                base64.b64encode(row["images"][0]["bytes"]).decode(),
                row["texts"][0]["user"].strip(),
                row["texts"][0]["assistant"].strip(),
                row,
            ))
        except Exception:
            continue

    out_rows = []
    n_kept = n_regen_grounding = n_dropped_low_q = n_nsfw = n_bad = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(process_one, b64, q, a): (b64, q, a, orig) for b64, q, a, orig in jobs}
        for fut in as_completed(futures):
            _, q, a, orig = futures[fut]
            try:
                res = fut.result()
            except Exception:
                n_bad += 1
                continue
            status = res["status"]
            if status == "kept":
                n_kept += 1
                # Preserve mmevol's native multi-turn structure
                out_rows.append({"images": orig["images"],
                                 "texts": orig["texts"],
                                 "source": orig["source"]})
            elif status == "regenerated_grounding":
                n_regen_grounding += 1
                out_rows.append({"images": orig["images"],
                                 "texts": res["texts"],  # 2-turn list from regen
                                 "source": orig["source"]})
            elif status == "dropped_low_quality":
                n_dropped_low_q += 1
            elif status == "nsfw_dropped":
                n_nsfw += 1
            else:
                n_bad += 1

    if out_rows:
        new_tbl = pa.Table.from_pylist(out_rows, schema=OUTPUT_SCHEMA)
        tmp = out_path.with_suffix(".parquet.tmp")
        pq.write_table(new_tbl, str(tmp), compression="zstd")
        tmp.rename(out_path)

    return {
        "shard": shard_path.name,
        "in": len(rows),
        "out": len(out_rows),
        "kept_original": n_kept,
        "regen_grounding": n_regen_grounding,
        "dropped_low_quality": n_dropped_low_q,
        "nsfw_dropped": n_nsfw,
        "bad": n_bad,
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
        shard_files = [f for f in shard_files
                       if int(re.search(r"train-(\d+)-of-", f.name).group(1)) in idxs]

    print(f"=== hybrid redistill mmevol: grounding-regen + non-grounding-judge ===")
    print(f"shards={len(shard_files)} concurrency={args.concurrency} threshold={PASS_THRESHOLD}\n", flush=True)
    t0 = time.time()
    total = {"in": 0, "out": 0, "kept": 0, "regen_g": 0, "drop_lq": 0, "nsfw": 0, "bad": 0, "skipped": 0}

    for sf in shard_files:
        out_path = OUT_ROOT / sf.name
        s0 = time.time()
        stats = process_shard(sf, out_path, args.concurrency)
        dt = time.time() - s0
        if stats.get("skipped"):
            print(f"  [skip] {sf.name}", flush=True); total["skipped"] += 1; continue
        if stats.get("empty"):
            print(f"  [empty] {sf.name}", flush=True); continue
        total["in"] += stats["in"]; total["out"] += stats["out"]
        total["kept"] += stats["kept_original"]
        total["regen_g"] += stats["regen_grounding"]
        total["drop_lq"] += stats["dropped_low_quality"]
        total["nsfw"] += stats["nsfw_dropped"]; total["bad"] += stats["bad"]
        print(f"  [{sf.name}] in={stats['in']:>5,} kept={stats['kept_original']:>5,} "
              f"regen_g={stats['regen_grounding']:>4,} drop_lq={stats['dropped_low_quality']:>4,} "
              f"nsfw={stats['nsfw_dropped']:>2} bad={stats['bad']:>3} "
              f"out={stats['out']:>5,} in {dt:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed:.0f}s ===")
    print(f"  in:                       {total['in']:,}")
    print(f"  kept original:            {total['kept']:,}  ({100*total['kept']/max(1,total['in']):.1f}%)")
    print(f"  regen (grounding fix):    {total['regen_g']:,}  ({100*total['regen_g']/max(1,total['in']):.1f}%)")
    print(f"  dropped (low quality):    {total['drop_lq']:,}  ({100*total['drop_lq']/max(1,total['in']):.1f}%)")
    print(f"  nsfw dropped:             {total['nsfw']:,}")
    print(f"  malformed dropped:        {total['bad']:,}")
    print(f"  total out:                {total['out']:,}  ({100*total['out']/max(1,total['in']):.1f}% retention)")


if __name__ == "__main__":
    main()
