"""Re-distill chinesememe with Qwen3.6-27B (Strategy H, 2-turn natural flow).

Pipeline:
  - read 29 shards from processed/sft/finevision/chinesememe/
  - for each shard:
      * spawn concurrent requests to Qwen router (default 100 in flight)
      * parse output: SAFETY tag, then **TURN 1** USER/ASSISTANT, then **TURN 2** USER/ASSISTANT
      * drop NSFW rows; drop unparseable rows (logged)
      * write new parquet to processed/sft/finevision/chinesememe_recap_en/{shard}
  - skip shards already present (idempotent)
  - schema matches FineVision: {images, texts: [{user,assistant}, ...], source}

Run:
    python redistill_chinesememe_qwen.py            # default settings
    python redistill_chinesememe_qwen.py --concurrency 200 --shards 0,1,2
"""

from __future__ import annotations
import argparse
import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

# ── config ────────────────────────────────────────────────────────────────────
ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"
IN_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/chinesememe")
OUT_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/chinesememe_recap_en")

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

# ── output parquet schema (matches FineVision shape) ──────────────────────────
OUTPUT_SCHEMA = pa.schema([
    ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
    ("texts", pa.list_(pa.struct([("user", pa.string()), ("assistant", pa.string())]))),
    ("source", pa.string()),
])


# ── Qwen client ───────────────────────────────────────────────────────────────
def call_qwen(img_b64: str, original_q: str, max_tokens: int = 1500, retries: int = 3) -> str | None:
    """Single inference call. Returns content string or None on failure."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": (
                    f"Original user question for this meme: \"{original_q}\"\n\n"
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


# ── parser ────────────────────────────────────────────────────────────────────
_TURN_RE = re.compile(
    r"\*\*TURN 1\*\*\s*\n"
    r"USER:\s*(.+?)\n"
    r"ASSISTANT:\s*(.+?)\n\s*"
    r"\*\*TURN 2\*\*\s*\n"
    r"USER:\s*(.+?)\n"
    r"ASSISTANT:\s*(.+?)\Z",
    re.DOTALL,
)


def parse_output(text: str) -> tuple[str, list[dict] | None]:
    """Returns (verdict, turns_or_None).

    verdict ∈ {"SAFE", "NSFW", "MALFORMED"}.
    If SAFE, returns 2-turn list. Otherwise returns None.
    """
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


# ── per-shard processing ──────────────────────────────────────────────────────
def process_shard(shard_path: Path, out_path: Path, concurrency: int) -> dict:
    """Process one shard: read → infer → parse → write. Returns stats dict."""
    if out_path.exists():
        return {"shard": shard_path.name, "skipped": True}

    tbl = pq.read_table(str(shard_path))
    rows = tbl.to_pylist()
    n_in = len(rows)
    if n_in == 0:
        return {"shard": shard_path.name, "empty": True}

    # Build payload list (image_b64, original_q, original_row)
    jobs = []
    for row in rows:
        try:
            img_bytes = row["images"][0]["bytes"]
            img_b64 = base64.b64encode(img_bytes).decode()
            original_q = row["texts"][0]["user"].strip()
            jobs.append((img_b64, original_q, row))
        except Exception:
            continue

    # Concurrent inference
    out_rows = []
    n_safe = n_nsfw = n_malformed = n_err = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(call_qwen, img_b64, q): (img_b64, q, orig)
            for img_b64, q, orig in jobs
        }
        for fut in as_completed(futures):
            _, _, orig = futures[fut]
            try:
                out = fut.result()
            except Exception:
                n_err += 1
                continue
            verdict, turns = parse_output(out)
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

    # Write output parquet (atomic via .tmp + rename)
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
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concurrency", type=int, default=100, help="Concurrent in-flight requests per shard")
    p.add_argument("--shards", type=str, default=None, help="Comma-sep shard indices to process (default: all)")
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(IN_ROOT.glob("train-*.parquet"))
    if args.shards:
        idxs = set(int(x) for x in args.shards.split(","))
        shard_files = [f for f in shard_files if int(re.search(r"train-(\d+)-of-", f.name).group(1)) in idxs]

    print(f"=== redistill chinesememe: {len(shard_files)} shards, "
          f"concurrency={args.concurrency}, router={ROUTER} ===\n", flush=True)
    t0 = time.time()
    total = {"in": 0, "out": 0, "safe": 0, "nsfw": 0, "malformed": 0, "errors": 0, "skipped": 0}

    for sf in shard_files:
        out_path = OUT_ROOT / sf.name
        s0 = time.time()
        stats = process_shard(sf, out_path, args.concurrency)
        dt = time.time() - s0
        if stats.get("skipped"):
            print(f"  [skip] {sf.name} (already exists)", flush=True)
            total["skipped"] += 1
            continue
        if stats.get("empty"):
            print(f"  [empty] {sf.name}", flush=True)
            continue
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
    print(f"  shards skipped:  {total['skipped']:,}")


if __name__ == "__main__":
    main()
