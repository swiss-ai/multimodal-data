"""Two checks, one pass over every parquet shard of every keep-list subset:

  (1) EXHAUSTIVE `<image>` literal scan
      - Apertus tokenizer maps the substring `<image>` (case-sensitive) to
        token 131079 = `<|image|>` (the special image placeholder)
      - Variants like `<IMAGE>`, `< image>`, `<Image>` are SAFE
      - Boundary attachments (`\\n<image>`, `<image>.`, etc.) also trigger
      - So: simple `'<image>' in text` substring search.

  (2) JACCARD DUPLICATE TURNS — assistant-side only, per multi-turn row
      - For each row with >= 2 turns, compute Jaccard similarity between
        every pair of assistant turn token-sets
      - A "dup" is a pair with Jaccard >= 0.8
      - Reports: % of multi-turn rows with at least one dup pair

Output:
  logs/image_dup_audit.json
"""
from __future__ import annotations
import os, sys, json, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

IN = "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision"
JACCARD_THRESHOLD = 0.8


def _tokenize(s: str) -> set:
    """Cheap word-token set (lowercase, no punctuation noise) for Jaccard."""
    return set(s.lower().split())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _scan_subset(subset: str) -> dict:
    in_dir = Path(IN) / subset
    if not (in_dir / ".complete").is_file():
        return {"subset": subset, "skipped": "no .complete marker"}

    files = sorted(in_dir.glob("train-*.parquet"))
    if not files:
        return {"subset": subset, "error": "no files"}

    total_rows = 0
    rows_with_image_literal = 0
    image_literal_in_user = 0
    image_literal_in_assistant = 0
    sample_image_examples = []

    multi_turn_rows = 0
    multi_turn_with_jaccard_dup = 0
    jaccard_pair_counts = Counter()  # bucketed by 0.1
    sample_dup_examples = []

    for f in files:
        try:
            tbl = pq.read_table(str(f), columns=["texts"])
        except Exception as e:
            return {"subset": subset, "error": f"read {f.name}: {e}"}

        for texts in tbl["texts"].to_pylist():
            total_rows += 1
            row_has_literal = False
            assistants = []
            for turn in texts:
                u = turn.get("user") or ""
                a = turn.get("assistant") or ""
                if "<image>" in u:
                    image_literal_in_user += 1
                    row_has_literal = True
                if "<image>" in a:
                    image_literal_in_assistant += 1
                    row_has_literal = True
                    if len(sample_image_examples) < 2:
                        snip = a[max(0, a.find("<image>")-40): a.find("<image>")+50]
                        sample_image_examples.append(snip)
                assistants.append(a)
            if row_has_literal:
                rows_with_image_literal += 1

            if len(assistants) >= 2:
                multi_turn_rows += 1
                # All pairwise Jaccards on assistant turns
                sets = [_tokenize(a) for a in assistants]
                max_j = 0.0
                worst_pair = None
                for i in range(len(sets)):
                    for j in range(i+1, len(sets)):
                        jc = _jaccard(sets[i], sets[j])
                        # bucket
                        bucket = round(jc, 1)
                        jaccard_pair_counts[bucket] += 1
                        if jc > max_j:
                            max_j = jc
                            worst_pair = (i, j)
                if max_j >= JACCARD_THRESHOLD:
                    multi_turn_with_jaccard_dup += 1
                    if len(sample_dup_examples) < 2 and max_j < 1.0:  # not exact-match for examples
                        i, j = worst_pair
                        sample_dup_examples.append({
                            "jaccard": max_j,
                            "ast_i": assistants[i][:120],
                            "ast_j": assistants[j][:120],
                        })

    return {
        "subset": subset,
        "total_rows": total_rows,
        "rows_with_image_literal": rows_with_image_literal,
        "image_literal_in_user": image_literal_in_user,
        "image_literal_in_assistant": image_literal_in_assistant,
        "image_literal_pct": 100*rows_with_image_literal/max(1,total_rows),
        "image_examples": sample_image_examples,
        "multi_turn_rows": multi_turn_rows,
        "multi_turn_dup_rows": multi_turn_with_jaccard_dup,
        "multi_turn_dup_pct": 100*multi_turn_with_jaccard_dup/max(1, multi_turn_rows),
        "jaccard_buckets": dict(sorted(jaccard_pair_counts.items())),
        "dup_examples": sample_dup_examples,
    }


def main():
    subsets = [
        p.name for p in sorted(Path(IN).iterdir())
        if (p / ".complete").is_file()
    ]
    print(f"=== scanning {len(subsets)} subsets ===", flush=True)
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(_scan_subset, s): s for s in subsets}
        for fut in as_completed(futures):
            r = fut.result()
            results[r["subset"]] = r
            if r.get("error"):
                print(f"[err] {r['subset']}: {r['error']}", flush=True)
            elif r.get("skipped"):
                print(f"[skip] {r['subset']}: {r['skipped']}", flush=True)
            else:
                img_flag = ""
                if r["rows_with_image_literal"] > 0:
                    img_flag = f" <image>={r['rows_with_image_literal']} ({r['image_literal_pct']:.3f}%)"
                dup_flag = ""
                if r["multi_turn_rows"] > 0 and r["multi_turn_dup_pct"] > 1:
                    dup_flag = f" jaccard_dup={r['multi_turn_dup_rows']}/{r['multi_turn_rows']} ({r['multi_turn_dup_pct']:.1f}%)"
                tag = (img_flag + dup_flag).strip() or "clean"
                print(f"  [{r['subset']:<28}] rows={r['total_rows']:>9,}  {tag}", flush=True)

    out = Path("/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/finevision/logs/image_dup_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\ndone in {time.time()-t0:.0f}s")
    print(f"full report: {out}")


if __name__ == "__main__":
    main()
