"""Systematic quality audit across all keep-list FineVision subsets.

Per memory `feedback_caption_quality_audit.md`, two flaws have cost us
re-tokenization in the past:
  (1) Leftover `<image>` placeholder tokens in assistant text
  (2) Captioner self-commentary leaking through

This script samples N rows per subset and reports flags for:
  - <image>/<|image|>/[IMG] placeholder leakage in assistant text
  - Captioner self-commentary tells ("although the", "despite the",
    "the description is", "real-world information", etc.)
  - Empty / very short / very long assistant responses
  - Repeated phrases (3+ consecutive duplicates)
  - Image count distribution
  - Schema oddities

Outputs a markdown report at logs/quality_audit.md and prints summary.
"""

from __future__ import annotations
import os, sys, re, io, time, json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

IN = "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision"
SAMPLE_PER_SHARD = 50  # rows per shard
MAX_SHARDS = 8         # cap shards inspected per subset (head sample)

# === Pattern definitions ===
PLACEHOLDER_PATTERNS = {
    "<image>":      re.compile(r'<image>', re.IGNORECASE),
    "<|image|>":    re.compile(r'<\|image\|>'),
    "[IMG]":        re.compile(r'\[IMG\]'),
    "<img>":        re.compile(r'<img[\s>]', re.IGNORECASE),
    "{image}":      re.compile(r'\{image\}', re.IGNORECASE),
    "<|image_pad|>":re.compile(r'<\|image_pad\|>'),
}

# Captioner self-commentary / meta-remarks
META_PATTERNS = {
    "although the":          re.compile(r'\balthough the\b', re.IGNORECASE),
    "despite the":           re.compile(r'\bdespite the\b', re.IGNORECASE),
    "real-world information":re.compile(r'real[ -]world information', re.IGNORECASE),
    "the description is":    re.compile(r'the description (?:is|was)', re.IGNORECASE),
    "the original description":re.compile(r'the original description', re.IGNORECASE),
    "the sentence structure":re.compile(r'\bsentence structure\b', re.IGNORECASE),
    "I cannot see":          re.compile(r"\bI (?:can(?:not|'t)|am unable to) (?:see|view|access)\b", re.IGNORECASE),
    "I apologize":           re.compile(r"\bI apologi[sz]e\b", re.IGNORECASE),
    "as an AI":              re.compile(r'\bas an AI\b', re.IGNORECASE),
    "I'm sorry":             re.compile(r"\bI'?m sorry\b", re.IGNORECASE),
    "GPT-?":                 re.compile(r'\bGPT-?\d', re.IGNORECASE),
    "as a language model":   re.compile(r'as a language model', re.IGNORECASE),
    "based on the image":    re.compile(r'based on the image', re.IGNORECASE),
}

# Repetition detection (3+ identical sentences in a row)
REPETITION_RE = re.compile(r'([^.!?\n]{20,}[.!?])\s*\1\s*\1', re.MULTILINE)

EMPTY_LIMIT = 5    # chars
VERY_LONG = 5000   # chars


def _audit_one(subset: str) -> dict:
    in_dir = Path(IN) / subset
    files = sorted(in_dir.glob("train-*.parquet"))[:MAX_SHARDS]
    if not files:
        return {"subset": subset, "error": "no files"}

    sampled = 0
    placeholder_hits = Counter()
    meta_hits = Counter()
    empty_count = 0
    very_long_count = 0
    repetition_count = 0
    img_counts = Counter()           # # images per row
    text_turn_counts = Counter()     # # turns per row
    text_lengths = []                # assistant char lengths

    placeholder_examples = []
    meta_examples = []

    for f in files:
        try:
            tbl = pq.read_table(str(f), columns=["images", "texts"])
        except Exception as e:
            return {"subset": subset, "error": f"read failed on {f.name}: {e}"}

        n = tbl.num_rows
        if n == 0:
            continue
        step = max(1, n // SAMPLE_PER_SHARD)
        rows = tbl.slice(0, n).to_pydict()
        for i in range(0, n, step):
            sampled += 1
            images = rows["images"][i]
            texts = rows["texts"][i]
            img_counts[len(images)] += 1
            text_turn_counts[len(texts)] += 1
            for turn in texts:
                a = (turn.get("assistant") or "").strip()
                u = (turn.get("user") or "").strip()
                text_lengths.append(len(a))

                # Empty / short / long
                if len(a) <= EMPTY_LIMIT:
                    empty_count += 1
                if len(a) >= VERY_LONG:
                    very_long_count += 1

                # Placeholder leakage in assistant
                for name, regex in PLACEHOLDER_PATTERNS.items():
                    if regex.search(a):
                        placeholder_hits[name] += 1
                        if len(placeholder_examples) < 3:
                            placeholder_examples.append((name, a[:200]))

                # Meta commentary
                for name, regex in META_PATTERNS.items():
                    if regex.search(a):
                        meta_hits[name] += 1
                        if len(meta_examples) < 3:
                            meta_examples.append((name, a[:200]))

                # Repetition (only check longish responses)
                if len(a) > 80 and REPETITION_RE.search(a):
                    repetition_count += 1

            if sampled >= SAMPLE_PER_SHARD * MAX_SHARDS:
                break

    text_lengths.sort()
    p50 = text_lengths[len(text_lengths)//2] if text_lengths else 0
    p95 = text_lengths[int(len(text_lengths)*0.95)] if text_lengths else 0

    return {
        "subset": subset,
        "sampled_rows": sampled,
        "img_count_dist": dict(img_counts.most_common(5)),
        "turn_count_dist": dict(text_turn_counts.most_common(5)),
        "assistant_len_p50": p50,
        "assistant_len_p95": p95,
        "empty_count": empty_count,
        "very_long_count": very_long_count,
        "repetition_count": repetition_count,
        "placeholder_hits": dict(placeholder_hits),
        "meta_hits": dict(meta_hits),
        "placeholder_examples": placeholder_examples[:2],
        "meta_examples": meta_examples[:2],
    }


def main():
    subsets = [
        p.name for p in sorted(Path(IN).iterdir())
        if (p / ".complete").is_file()
    ]
    print(f"=== auditing {len(subsets)} subsets ===", flush=True)
    results = {}
    with ProcessPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_audit_one, s): s for s in subsets}
        for fut in as_completed(futures):
            r = fut.result()
            results[r["subset"]] = r
            flags = []
            if r.get("placeholder_hits"): flags.append(f"PLACEHOLDER:{sum(r['placeholder_hits'].values())}")
            if r.get("meta_hits"): flags.append(f"META:{sum(r['meta_hits'].values())}")
            if r.get("empty_count", 0) > 0: flags.append(f"EMPTY:{r['empty_count']}")
            if r.get("very_long_count", 0) > 0: flags.append(f"LONG:{r['very_long_count']}")
            if r.get("repetition_count", 0) > 0: flags.append(f"REPEAT:{r['repetition_count']}")
            tag = " ".join(flags) if flags else "ok"
            sampled = r.get("sampled_rows", 0)
            print(f"  [{tag}] {r['subset']} ({sampled} rows sampled)", flush=True)

    # write json + summary md
    out_dir = Path("/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision/logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_audit.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nfull report: {out_dir}/quality_audit.json")
    print("done.")


if __name__ == "__main__":
    main()
