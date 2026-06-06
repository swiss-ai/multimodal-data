"""Full SpatialSense redistill: Qwen3.6-27B with bbox-grounded answers + filter.

Pipeline:
  1. For each row, ask Qwen (image + question, NOT told the gold label) to:
       - Localize each named object with a bounding box
       - State Yes/No + 1-sentence grounded spatial relation
  2. Parse Qwen's Yes/No from output
  3. If matches gold (the human-labeled SpatialSense binary): KEEP, write the
     bbox-grounded answer to spatialsense_recap/
  4. If contradicts: DROP — Qwen identified a different object pair than the
     human annotator did. SpatialSense is the gold (adversarially crowdsourced
     by Princeton VL), so trust it.

Output preserves the BLIP-style grounding format that mmevol_judged uses:
  <object>NAME</object><bbox>[X1,Y1,X2,Y2]</bbox> — coords normalized 0-1000.

Fan-out: 32 concurrent requests round-robin across worker URLs read from
/tmp/ready_workers.txt. Idempotent at shard level.
"""

from __future__ import annotations
import base64
import glob
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import requests


SRC_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/spatialsense")
OUT_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/spatialsense_recap")
WORKERS_FILE = Path("/tmp/ready_workers.txt")
MODEL = "Qwen/Qwen3.6-27B-xyixuan"
CONCURRENCY = 32
TIMEOUT = 60
RETRIES = 2
SAMPLE_EVERY = 200


# v5 — INTEGRATED: answer first, then single grounded explanation with inline
# bboxes; explicit 3D-vs-2D reasoning guidance (overlap in 2D may be in front
# of in 3D); independent Yes/No judgment for gold-agreement filtering.
SYSTEM_PROMPT = """You answer a Yes/No spatial relation question about an image. You reason about the scene in 3D — not just 2D bbox geometry.

Output format (one self-contained line):

<answer>Yes</answer>. <description>One sentence describing the ACTUAL 3D spatial relation, with each named object inline-grounded as <object>NAME</object><bbox>[X1,Y1,X2,Y2]</bbox>.</description>

OR

<answer>No</answer>. <description>One sentence stating the ACTUAL 3D relation (do NOT just negate the asked relation), with each named object inline-grounded.</description>

Rules:
  1. Begin with <answer>Yes</answer>. or <answer>No</answer>. — judge from the image, not from the wording of the question.
  2. Each named object appears INLINE inside the description as <object>NAME</object><bbox>[X_min, Y_min, X_max, Y_max]</bbox>. Coords are normalized 0-1000 with the top-left as origin.
  3. Reason in 3D using ALL visual cues — depth, occlusion (which object blocks which), apparent size, perspective, gravity. Do NOT infer relations from 2D bbox overlap alone.
       * "behind" / "in front of" = 3D depth (closer to or farther from the camera). Two bboxes can overlap in 2D yet one object is in front of the other in 3D.
       * "above" / "below" = world-vertical position (gravity-aligned), often inferable from the scene.
       * "on top of" / "under" = direct vertical support contact, not just 2D y-overlap.
  4. Use a concrete relation term from {above, below, to the left of, to the right of, behind, in front of, on top of, under, beside, next to, inside, outside, between, near}.
  5. NEVER use: "elsewhere", "different relationship", "appears to", "seems to", "may be", "possibly".
  6. If the scene has multiple instances of a named object, pick the most salient and clearly visible pair.

Examples:

Example 1 (Yes — straightforward):
<answer>Yes</answer>. <description>The <object>cat</object><bbox>[180,420,540,760]</bbox> is on top of the <object>ground</object><bbox>[0,600,1000,1000]</bbox>, lying on a brown tile floor near a mirror.</description>

Example 2 (No — 3D depth, NOT just 2D positioning):
<answer>No</answer>. <description>The <object>lizard</object><bbox>[340,510,620,690]</bbox> is in front of the <object>tree</object><bbox>[450,100,700,950]</bbox>, on the dirt closer to the camera (even though their 2D bboxes partially overlap, the lizard occludes the tree's trunk).</description>

Example 3 (No — 3D occlusion):
<answer>No</answer>. <description>The <object>window</object><bbox>[200,100,600,500]</bbox> is behind the <object>man</object><bbox>[300,200,500,800]</bbox>, visible through the glass panes in the background of the scene.</description>

Example 4 (Yes — true 3D support contact):
<answer>Yes</answer>. <description>The <object>book</object><bbox>[120,440,580,580]</bbox> is on top of the <object>shelf</object><bbox>[100,420,900,520]</bbox>, sitting flush on the wooden surface between two other books.</description>"""


# Parsers
RE_ANSWER = re.compile(r"<answer>\s*(Yes|No)\s*</answer>", re.IGNORECASE)
RE_BBOX = re.compile(r"<object>([^<]+)</object>\s*<bbox>\s*\[([^\]]+)\]\s*</bbox>", re.IGNORECASE)
# v5 description contains inline <object>/<bbox> tags, so we cannot use [^<]+.
RE_DESC = re.compile(r"<description>(.*?)</description>", re.IGNORECASE | re.DOTALL)


def load_workers() -> list[str]:
    return [u.strip() for u in WORKERS_FILE.read_text().splitlines() if u.strip()]


def to_data_url(image_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"


def extract_gold_label(answer_text: str) -> str:
    head = answer_text.strip()[:4].lower()
    if head.startswith("yes"): return "Yes"
    if head.startswith("no"):  return "No"
    return ""


def parse_qwen_output(text: str) -> dict | None:
    """Return {label, bboxes:[(name, [X1,Y1,X2,Y2])], description_with_inline_bboxes}
    or None on parse fail.

    v5 format: bboxes are INLINE inside <description>...</description>; the
    description is the integrated single sentence we want to emit verbatim.
    """
    am = RE_ANSWER.search(text)
    if not am:
        return None
    label = am.group(1).capitalize()
    dm = RE_DESC.search(text)
    if not dm:
        return None
    desc = dm.group(1).strip()
    if not desc:
        return None
    # Pull inline bboxes from within the description for the contradiction-filter
    # and for downstream validation; the description string itself is what we emit.
    bboxes = []
    for m in RE_BBOX.finditer(desc):
        name = m.group(1).strip()
        try:
            coords = [int(round(float(c.strip()))) for c in m.group(2).split(",")]
        except ValueError:
            continue
        if len(coords) == 4:
            # Clamp coords to 0-1000 in case Qwen overshoots
            coords = [max(0, min(1000, c)) for c in coords]
            bboxes.append((name, coords))
    if len(bboxes) < 2:
        return None
    return {"label": label, "bboxes": bboxes, "description": desc}


def format_grounded_answer(parsed: dict) -> str:
    """Integrated format: '<Label>. <description with inline bboxes>'"""
    return f"{parsed['label']}. {parsed['description']}"


def call_qwen(worker_url: str, image_data_url: str, question: str) -> str | None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": question},
            ]},
        ],
        "max_tokens": 280,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(f"{worker_url}/v1/chat/completions", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            c = (msg.get("content") or msg.get("reasoning") or "").strip()
            if c:
                return c
        except Exception:
            if attempt == RETRIES:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def redistill_row(row: dict, workers: list[str], worker_idx: int) -> tuple[dict | None, int, int]:
    """Per-TURN drop: keep good turns, drop only the turns that fail parse or
    contradict gold. Returns (new_row | None, turns_kept, turns_dropped).
    Row is None only if ZERO turns survived."""
    new_turns = []
    n_dropped = 0
    img_url = to_data_url(row["images"][0]["bytes"])
    for turn in (row.get("texts") or []):
        u_raw = (turn.get("user") or "")
        u = u_raw.replace("<image>\n", "").strip()
        gold_label = extract_gold_label(turn.get("assistant") or "")
        if not gold_label:
            n_dropped += 1
            continue

        worker = workers[worker_idx % len(workers)]
        worker_idx += 1
        out = call_qwen(worker, img_url, u)
        if out is None:
            n_dropped += 1
            continue
        parsed = parse_qwen_output(out)
        if parsed is None:
            n_dropped += 1
            continue
        if parsed["label"] != gold_label:
            n_dropped += 1  # contradiction: drop this turn only
            continue

        new_turns.append({"user": u_raw, "assistant": format_grounded_answer(parsed)})

    if not new_turns:
        return None, 0, n_dropped
    out = dict(row)
    out["texts"] = new_turns
    return out, len(new_turns), n_dropped


def process_shard(shard_path: Path, out_path: Path, workers: list[str]) -> dict:
    tbl = pq.read_table(str(shard_path))
    rows = tbl.to_pylist()
    n_in = len(rows)
    kept: list[dict | None] = [None] * n_in
    base_idx = random.randint(0, 1_000_000)

    turn_stats = {"kept": 0, "dropped": 0}

    def task(i: int, row: dict):
        return i, redistill_row(row, workers, base_idx + i * 7)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(task, i, r) for i, r in enumerate(rows)]
        done = 0
        for f in as_completed(futs):
            i, (new_row, n_kept_turns, n_dropped_turns) = f.result()
            kept[i] = new_row
            turn_stats["kept"] += n_kept_turns
            turn_stats["dropped"] += n_dropped_turns
            done += 1
            if done % SAMPLE_EVERY == 0:
                samples_kept = [r for r in kept[:done] if r is not None]
                if samples_kept:
                    s = random.choice(samples_kept)
                    t0 = s["texts"][0]
                    sys.stderr.write(
                        f"  [QC @ {done}/{n_in}, rows-kept={len(samples_kept)}, "
                        f"turns kept={turn_stats['kept']:,} dropped={turn_stats['dropped']:,}] "
                        f"Q={t0['user'][:80]!r} A={t0['assistant'][:200]!r}\n")

    kept_rows = [r for r in kept if r is not None]
    final_tbl = pa.Table.from_pylist(kept_rows, schema=tbl.schema)
    tmp = out_path.with_suffix(".parquet.tmp")
    pq.write_table(final_tbl, str(tmp), compression="zstd")
    tmp.rename(out_path)
    return {"in": n_in, "kept": len(kept_rows), "dropped": n_in - len(kept_rows),
            "turns_kept": turn_stats["kept"], "turns_dropped": turn_stats["dropped"]}


def main():
    workers = load_workers()
    if not workers:
        sys.stderr.write("no workers in /tmp/ready_workers.txt\n")
        sys.exit(1)
    print(f"workers ({len(workers)}): {workers}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(SRC_DIR.glob("train-*.parquet"))
    print(f"=== {len(shards)} shards to process ===", flush=True)

    t0 = time.time()
    total_in = total_kept = total_turns_kept = total_turns_dropped = 0
    for sh in shards:
        out_path = OUT_DIR / sh.name
        if out_path.exists() and out_path.stat().st_size > 0:
            n = pq.read_metadata(out_path).num_rows
            print(f"  [{sh.name}] SKIP (already done, {n:,} rows)", flush=True)
            total_kept += n
            continue
        t_shard = time.time()
        stats = process_shard(sh, out_path, workers)
        dt = time.time() - t_shard
        total_in += stats["in"]
        total_kept += stats["kept"]
        total_turns_kept += stats["turns_kept"]
        total_turns_dropped += stats["turns_dropped"]
        rate = stats["in"] / dt
        print(f"  [{sh.name}] rows in={stats['in']:>5,} kept={stats['kept']:>5,} "
              f"({100*stats['kept']/stats['in']:.1f}%)  "
              f"turns kept={stats['turns_kept']:>5,} dropped={stats['turns_dropped']:>5,}  "
              f"in {dt:.0f}s ({rate:.1f} rows/s)", flush=True)

    print(f"\n=== TOTAL ===")
    print(f"  rows in:        {total_in:,}")
    print(f"  rows kept:      {total_kept:,}  ({100*total_kept/max(1,total_in):.1f}%)")
    print(f"  turns kept:     {total_turns_kept:,}")
    print(f"  turns dropped:  {total_turns_dropped:,}  "
          f"({100*total_turns_dropped/max(1,total_turns_kept+total_turns_dropped):.1f}%)")
    print(f"  elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
