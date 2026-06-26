"""Re-caption localized_narratives with Qwen3.6-27B.

Original LN captions are human-spoken narrations transcribed mostly vague:
"some persons standing at right side", "group of people", "many furnitures".
Qwen3.6-27B can produce more precise, count-aware descriptions.

Two modes:
  --mode=full     re-caption ALL rows (181K)
  --mode=vague    re-caption ONLY the ~49K rows with vague-but-no-precise patterns
                  (default; the "rescue vague" use case)

Output: 1-turn (same shape as LN), single rich English caption.

Run:
    python redistill_localized_narratives_qwen.py --mode=vague --concurrency 100
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
IN_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/localized_narratives")
OUT_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/localized_narratives_recap")

# Patterns for "vague" filtering (same as audit script)
VAGUE_RE = re.compile(
    r'\b(a group of|groups of|some \w+|several|a few|a couple of|many|lots of|plenty of|numerous|various)\b',
    re.IGNORECASE,
)
PRECISE_RE = re.compile(
    r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred|thousand|\d+)\b',
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You produce precise, factual English captions of images for vision-language "
    "model training. Follow this exact protocol.\n\n"
    "STEP 1 — Safety check. Output one of:\n"
    "  **SAFETY: SAFE**\n"
    "  **SAFETY: NSFW**\n\n"
    "Mark NSFW only for: sexual / nudity / sexualized minors, graphic violence "
    "or gore, hate speech or slurs, self-harm or suicide content, illegal drug "
    "glorification, real-person harassment. Ordinary scenes are SAFE.\n\n"
    "STEP 2 — If NSFW, write one-sentence reason and STOP.\n\n"
    "STEP 3 — If SAFE, generate a precise English caption of the image as ONE "
    "paragraph (60-150 words). Rules:\n"
    "  - State EXACT COUNTS for people, animals, objects when small enough to "
    "count (1-10). Use vague qualifiers (many, several, a few) only when count "
    "truly cannot be determined.\n"
    "  - Describe spatial relationships (left/right, foreground/background, "
    "above/below).\n"
    "  - Name specific objects, colors, materials, actions, expressions. Avoid "
    "fluff like 'beautiful', 'amazing', 'wonderful'.\n"
    "  - Single paragraph, neutral declarative tone — like a museum label.\n\n"
    "Output format:\n"
    "**SAFETY: SAFE**\n\n"
    "**Caption:**\n"
    "<the caption paragraph>\n"
)

OUTPUT_SCHEMA = pa.schema([
    ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    ("texts", pa.list_(pa.struct([("user", pa.string()), ("assistant", pa.string())]))),
    ("source", pa.string()),
])


def call_qwen(img_b64: str, q: str, max_tokens: int = 600, retries: int = 3) -> str | None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": q},
            ]},
        ],
        "temperature": 0.2,
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


_CAP_RE = re.compile(r"\*\*Caption:\*\*\s*\n(.+?)\Z", re.DOTALL)


def parse_output(text: str) -> tuple[str, str | None]:
    if not text:
        return "MALFORMED", None
    if "SAFETY: NSFW" in text:
        return "NSFW", None
    if "SAFETY: SAFE" not in text:
        return "MALFORMED", None
    m = _CAP_RE.search(text)
    if not m:
        return "MALFORMED", None
    return "SAFE", m.group(1).strip()


def is_target_row(assistant_text: str, mode: str) -> bool:
    """Decide if a row qualifies for re-caption under the chosen mode."""
    if mode == "full":
        return True
    # mode = "vague": only rows that have vague counts WITHOUT any precise number
    has_vague = bool(VAGUE_RE.search(assistant_text))
    has_precise = bool(PRECISE_RE.search(assistant_text))
    return has_vague and not has_precise


def process_shard(shard_path: Path, out_path: Path, concurrency: int, mode: str) -> dict:
    if out_path.exists():
        return {"shard": shard_path.name, "skipped": True}

    tbl = pq.read_table(str(shard_path))
    rows = tbl.to_pylist()
    n_in = len(rows)
    if n_in == 0:
        return {"shard": shard_path.name, "empty": True}

    jobs = []
    n_skipped_nontarget = 0
    for row in rows:
        try:
            a = row["texts"][0]["assistant"]
            if not is_target_row(a, mode):
                n_skipped_nontarget += 1
                continue
            img_bytes = row["images"][0]["bytes"]
            img_b64 = base64.b64encode(img_bytes).decode()
            q = row["texts"][0]["user"].strip()
            jobs.append((img_b64, q, row))
        except Exception:
            continue

    if not jobs:
        return {"shard": shard_path.name, "in": n_in, "out": 0, "safe": 0, "nsfw": 0,
                "malformed": 0, "errors": 0, "skipped_nontarget": n_skipped_nontarget}

    out_rows = []
    n_safe = n_nsfw = n_malformed = n_err = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(call_qwen, b64, q): (b64, q, orig) for b64, q, orig in jobs}
        for fut in as_completed(futures):
            _, q, orig = futures[fut]
            try:
                out = fut.result()
            except Exception:
                n_err += 1
                continue
            verdict, caption = parse_output(out)
            if verdict == "SAFE":
                n_safe += 1
                out_rows.append({
                    "images": orig["images"],
                    "texts": [{"user": q, "assistant": caption}],
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
        "in": n_in,
        "out": len(out_rows),
        "safe": n_safe,
        "nsfw": n_nsfw,
        "malformed": n_malformed,
        "errors": n_err,
        "skipped_nontarget": n_skipped_nontarget,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["vague", "full"], default="vague",
                   help="vague = rescue ~49K vague-only rows; full = recap all 181K")
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--shards", type=str, default=None)
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(IN_ROOT.glob("train-*.parquet"))
    if args.shards:
        idxs = set(int(x) for x in args.shards.split(","))
        shard_files = [f for f in shard_files if int(re.search(r"train-(\d+)-of-", f.name).group(1)) in idxs]

    print(f"=== redistill localized_narratives: mode={args.mode}, {len(shard_files)} shards, "
          f"concurrency={args.concurrency} ===\n", flush=True)
    t0 = time.time()
    total = {"in": 0, "out": 0, "safe": 0, "nsfw": 0, "malformed": 0, "errors": 0,
             "skipped_nontarget": 0, "skipped_shards": 0}

    for sf in shard_files:
        out_path = OUT_ROOT / sf.name
        s0 = time.time()
        stats = process_shard(sf, out_path, args.concurrency, args.mode)
        dt = time.time() - s0
        if stats.get("skipped"):
            print(f"  [skip] {sf.name}", flush=True); total["skipped_shards"] += 1; continue
        if stats.get("empty"):
            print(f"  [empty] {sf.name}", flush=True); continue
        for k in ("in", "out", "safe", "nsfw", "malformed", "errors", "skipped_nontarget"):
            total[k] += stats.get(k, 0)
        targeted = stats["in"] - stats.get("skipped_nontarget", 0)
        kept = 100 * stats["out"] / max(1, targeted)
        print(f"  [{sf.name}] in={stats['in']:>5,} targeted={targeted:>5,} "
              f"out={stats['out']:>5,} ({kept:>5.1f}% of targeted kept) "
              f"safe={stats['safe']} nsfw={stats['nsfw']} malformed={stats['malformed']} "
              f"err={stats['errors']} in {dt:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== done in {elapsed:.0f}s ===")
    print(f"  total input rows:       {total['in']:,}")
    print(f"  skipped (non-target):   {total['skipped_nontarget']:,}")
    print(f"  targeted:               {total['in'] - total['skipped_nontarget']:,}")
    print(f"  output (safe):          {total['safe']:,}")
    print(f"  nsfw filtered:          {total['nsfw']:,}")
    print(f"  malformed:              {total['malformed']:,}")
    print(f"  request errors:         {total['errors']:,}")


if __name__ == "__main__":
    main()
