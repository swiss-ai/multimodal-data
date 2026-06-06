"""Quality-assess a FineVision subset using Qwen3.6-27B as judge.

For each sampled row, Qwen sees (image, user_question, original_assistant_answer)
and scores it on 4 dimensions (1-5 Likert):
  - Factual correctness
  - Visual grounding accuracy
  - Reasoning quality
  - Format quality

Aggregate scores tell us whether the subset is worth keeping / re-distilling.

Run:
    python quality_assess_subset.py --subset mmevol --sample 50
    python quality_assess_subset.py --subset chinesememe --sample 30
"""

from __future__ import annotations
import argparse
import base64
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import requests

ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"
ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision")

JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of vision-language model training data. "
    "Given an image, a user question, and an assistant answer, score the answer "
    "on FOUR dimensions using a 1-5 Likert scale (5 = excellent, 1 = poor).\n\n"
    "DIMENSIONS:\n"
    "  1. **Factual correctness**: Does the answer correctly address the question "
    "given what's in the image? (1=wrong, 3=partially right, 5=fully correct)\n"
    "  2. **Visual grounding**: Does the assistant's references to objects, "
    "locations, or visual elements match what's actually in the image? "
    "(1=hallucinated, 3=mixed, 5=fully grounded)\n"
    "  3. **Reasoning quality**: Is the explanation/reasoning sound, well-structured, "
    "and pedagogically useful? (1=incoherent, 3=adequate, 5=excellent)\n"
    "  4. **Format quality**: Is the output clean, consistent, readable, free of "
    "garbled text or weird formatting? (1=broken, 3=mostly clean, 5=polished)\n\n"
    "Output EXACTLY this format (numeric scores only, no decimals):\n\n"
    "**Factual:** <1-5>\n"
    "**Grounding:** <1-5>\n"
    "**Reasoning:** <1-5>\n"
    "**Format:** <1-5>\n"
    "**Issues:** <one-sentence flag of any specific problem, or 'none'>"
)


def call_qwen(img_b64: str, question: str, answer: str, max_tokens: int = 300) -> str | None:
    user_text = (
        f"USER QUESTION: {question}\n\n"
        f"ASSISTANT ANSWER (to evaluate):\n{answer[:2500]}\n\n"
        f"Score this answer per the protocol."
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": user_text},
            ]},
        ],
        "temperature": 0.1,  # low temp for judging consistency
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(3):
        try:
            r = requests.post(f"{ROUTER}/v1/chat/completions", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 * (attempt + 1))


_SCORE_RE = {
    "factual":   re.compile(r"\*\*Factual:\*\*\s*(\d)"),
    "grounding": re.compile(r"\*\*Grounding:\*\*\s*(\d)"),
    "reasoning": re.compile(r"\*\*Reasoning:\*\*\s*(\d)"),
    "format":    re.compile(r"\*\*Format:\*\*\s*(\d)"),
}
_ISSUES_RE = re.compile(r"\*\*Issues:\*\*\s*(.+?)(?:\n|\Z)", re.DOTALL)


def parse(text: str) -> dict | None:
    if not text:
        return None
    scores = {}
    for k, rx in _SCORE_RE.items():
        m = rx.search(text)
        if not m:
            return None
        scores[k] = int(m.group(1))
    m = _ISSUES_RE.search(text)
    scores["issues"] = m.group(1).strip() if m else ""
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", required=True, help="subset name under processed/sft/finevision/")
    p.add_argument("--sample", type=int, default=50, help="number of rows to assess")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    subset_dir = ROOT / args.subset
    if not subset_dir.is_dir():
        print(f"ERROR: {subset_dir} not found")
        return 1

    shards = sorted(subset_dir.glob("train-*.parquet"))
    print(f"=== quality assess: {args.subset} ({len(shards)} shards) ===")
    print(f"sampling {args.sample} rows with seed={args.seed}\n")

    # Load all rows then sample (small subsets only; or pick from first few shards)
    # For efficiency, just sample from first 3 shards
    rows = []
    for f in shards[:3]:
        tbl = pq.read_table(str(f), columns=["images", "texts", "source"])
        rows.extend(tbl.to_pylist())
        if len(rows) >= max(5000, args.sample * 50):
            break
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.sample]
    print(f"sampled {len(rows)} rows from first {min(len(shards), 3)} shards\n")

    # Build jobs: (img_b64, question, answer, idx)
    jobs = []
    for idx, row in enumerate(rows):
        try:
            img_b64 = base64.b64encode(row["images"][0]["bytes"]).decode()
            q = row["texts"][0]["user"].strip()
            a = row["texts"][0]["assistant"].strip()
            jobs.append((idx, img_b64, q, a))
        except Exception:
            continue

    # Concurrent judge calls
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(call_qwen, b64, q, a): (idx, q, a) for idx, b64, q, a in jobs}
        for fut in as_completed(futures):
            idx, q, a = futures[fut]
            try:
                raw = fut.result()
            except Exception:
                continue
            scores = parse(raw)
            if scores:
                results.append((idx, scores))

    elapsed = time.time() - t0
    print(f"=== {len(results)}/{len(jobs)} valid responses in {elapsed:.0f}s ===\n")

    if not results:
        print("no valid scores parsed")
        return 1

    # Aggregate stats
    dims = ["factual", "grounding", "reasoning", "format"]
    print(f"{'dim':<12} {'mean':>6} {'p25':>6} {'p50':>6} {'p75':>6} {'n_1or2':>8}")
    print("-" * 50)
    for d in dims:
        vals = sorted(s[d] for _, s in results)
        n = len(vals)
        mean = sum(vals) / n
        p25 = vals[n // 4]
        p50 = vals[n // 2]
        p75 = vals[3 * n // 4]
        n_bad = sum(1 for v in vals if v <= 2)
        print(f"{d:<12} {mean:>6.2f} {p25:>6} {p50:>6} {p75:>6} {n_bad:>4} ({100*n_bad/n:.0f}%)")

    # Top 5 worst rows for inspection
    print(f"\n=== top 5 worst (lowest combined score) ===")
    scored = [(idx, s, sum(s[d] for d in dims)) for idx, s in results]
    scored.sort(key=lambda x: x[2])
    for idx, s, tot in scored[:5]:
        print(f"  row {idx}: total={tot}/20  F={s['factual']} G={s['grounding']} "
              f"R={s['reasoning']} Fmt={s['format']}  | issues: {s['issues'][:120]}")

    # Save full results to json
    out_path = Path(f"/tmp/quality_assess_{args.subset}.json")
    out_path.write_text(json.dumps([{"idx": i, **s} for i, s in results], indent=2))
    print(f"\nfull scores: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
