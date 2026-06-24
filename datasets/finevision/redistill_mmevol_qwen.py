"""Re-distill mmevol with Qwen3.6-27B (Strategy I, reasoning + BLIP grounding).

mmevol is GPT-4V-distilled visual reasoning with embedded 0-1 float bbox coords.
We regenerate with Qwen3.6-27B (≈+14 MMMU pts over GPT-4V) and convert coords
to Apertus's BLIP-style <object>NAME</object><bbox>[X,Y,X,Y]</bbox> with 0-1000 ints.

Strategy F's biggest failure was bbox format compliance — Qwen kept emitting
`<bbox>...` without the `<object>NAME</object>` wrapper, and bracket placement
was inconsistent. Fix: a CONCRETE EXAMPLE inside the system prompt showing the
exact format. Few-shot >>> instruction for non-native formats.

Output: 1-turn (the original mmevol shape), single rich answer with reasoning +
grounding + answer. Multi-turn evolution rephrasings dropped.

Run:
    python redistill_mmevol_qwen.py             # default
    python redistill_mmevol_qwen.py --shards 0,1 --concurrency 100
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

# ── config ────────────────────────────────────────────────────────────────────
ROUTER = Path("/tmp/qwen_router.url").read_text().strip()
MODEL = "Qwen/Qwen3.6-27B-smatrenok"
IN_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/mmevol")
OUT_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/mmevol_regrounded")

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
                {"type": "text", "text": q},
            ]},
        ],
        "temperature": 0.3,
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


_RESP_RE = re.compile(
    r"\*\*Reasoning:\*\*\s*\n(.+?)\n\s*\*\*Answer:\*\*\s*\n(.+?)\Z",
    re.DOTALL,
)


def parse_output(text: str) -> tuple[str, str | None]:
    """Returns (verdict, assistant_text_or_None).

    On SAFE: assistant_text combines reasoning + answer (preserving format
    so the bbox grounding is in the supervision).
    """
    if not text:
        return "MALFORMED", None
    if "SAFETY: NSFW" in text:
        return "NSFW", None
    if "SAFETY: SAFE" not in text:
        return "MALFORMED", None
    m = _RESP_RE.search(text)
    if not m:
        return "MALFORMED", None
    reasoning = m.group(1).strip()
    answer = m.group(2).strip()
    # Combined assistant output keeps both sections in the same format the
    # student model is expected to emit.
    combined = f"**Reasoning:**\n{reasoning}\n\n**Answer:**\n{answer}"
    return "SAFE", combined


def process_shard(shard_path: Path, out_path: Path, concurrency: int) -> dict:
    if out_path.exists():
        return {"shard": shard_path.name, "skipped": True}

    tbl = pq.read_table(str(shard_path))
    rows = tbl.to_pylist()
    n_in = len(rows)
    if n_in == 0:
        return {"shard": shard_path.name, "empty": True}

    jobs = []
    for row in rows:
        try:
            img_bytes = row["images"][0]["bytes"]
            img_b64 = base64.b64encode(img_bytes).decode()
            # Use turn-0 question (most elaborate; subsequent turns are rephrasings)
            q = row["texts"][0]["user"].strip()
            jobs.append((img_b64, q, row))
        except Exception:
            continue

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
            verdict, assistant = parse_output(out)
            if verdict == "SAFE":
                n_safe += 1
                out_rows.append({
                    "images": orig["images"],
                    "texts": [{"user": q, "assistant": assistant}],
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

    print(f"=== redistill mmevol: {len(shard_files)} shards, "
          f"concurrency={args.concurrency}, router={ROUTER} ===\n", flush=True)
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
