"""lnqa redistill: keep the original questions, regenerate assistant answers
with Qwen3.6-27B grounded in the image.

Why: FineVision's lnqa answers are template-driven single-fact responses that
echo the question with minimal info ("There is X in the image."). We replace
each assistant message with a richer, image-grounded response (2-4 sentences,
specific visual detail, named-entity mentions). The Qs are preserved as-is.

License: CC-BY-4.0 (commercial OK).

Fan-out: round-robin over /tmp/ready_workers.txt with 32 concurrency.
Idempotent at shard level.
"""

from __future__ import annotations
import base64
import glob
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import requests


SRC_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/finevision/lnqa")
OUT_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/lnqa_recap")
WORKERS_FILE = Path("/tmp/ready_workers.txt")
# Filesystem-published endpoints from the proven serve_qwen36_40n.slurm pattern
ENDPOINTS_DIR_SYMLINK = Path("/iopsstor/scratch/cscs/xyixuan/apertus/benchmark-image-tokenzier/logs/qwen36_endpoints_latest")
MODEL = "Qwen/Qwen3.6-27B-xyixuan"
CONCURRENCY = 400
TIMEOUT = 60
RETRIES = 2
SAMPLE_EVERY = 100
MAX_ROWS_PER_SHARD = None  # None = all rows; set e.g. 200 for sub-sampling test runs


SYSTEM_PROMPT = """You answer a question about an image with a SPECIFIC, GROUNDED response. The questions you receive are often generic ("What is in the middle of the image?", "What can be seen on the road?"). Your job is to answer with rich visual detail.

Rules:
  1. 2-4 sentences. Natural English.
  2. Be SPECIFIC about what you see — name objects, colors, positions, materials, actions, count where relevant.
  3. If the question is template-y (e.g. "What is visible at the top?"), still give a detailed answer that actually describes what's there in the image, not a one-word echo.
  4. If the question is nonsensical for an image (e.g. "What does the taste of the rock feel like?"), answer briefly and pivot to a relevant visual fact about the named object: "The rock in the image is grey and weathered, sitting on a forest floor — taste isn't something visible in a still image."
  5. If the question assumes something not in the image (e.g. asks about a sofa when none is shown), say so clearly and describe what IS in that region of the image.
  6. NEVER use empty phrases like "There is X in the image." as a complete answer. NEVER use "appears to be", "seems to be", "may be", "possibly".
  7. Do NOT prefix with "Yes" / "No" or restate the question.

Examples:

Q: What is the main subject in the middle of the image?
A: A snow-capped mountain range dominates the middle of the image, with deep blue alpine water in the foreground reflecting the peaks. A small wooden cabin sits on the left shore, partially obscured by pine trees.

Q: What type of vegetation is present in the image?
A: Small flowering plants — likely wildflowers — cluster along the path's edge, with their pale violet blooms standing out against the brown gravel. Behind them, a row of low shrubs leads up to a stand of birch trees.

Q: What can be seen reflecting on the glass in the image?
A: A small toy figurine — appears to be a plastic dinosaur — is reflected on the window glass, doubled against the streetlights visible through the pane. The reflection slightly distorts the toy's outline due to the curvature of the window."""


def load_workers() -> list[str]:
    """Load worker URLs.

    Preference order:
      1. ENDPOINTS_DIR_SYMLINK/*.endpoint (filesystem-published from
         serve_qwen36_40n.slurm) — one host:port per file.
      2. WORKERS_FILE — legacy /tmp/ready_workers.txt (one URL per line).
    """
    if ENDPOINTS_DIR_SYMLINK.is_dir():
        eps = sorted(ENDPOINTS_DIR_SYMLINK.glob("*.endpoint"))
        urls = []
        for f in eps:
            hostport = f.read_text().strip()
            if hostport and not hostport.startswith("http"):
                urls.append(f"http://{hostport}")
            elif hostport:
                urls.append(hostport)
        if urls:
            return urls
    return [u.strip() for u in WORKERS_FILE.read_text().splitlines() if u.strip()]


def to_data_url(image_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(image_bytes).decode()}"


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
        "max_tokens": 240,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(f"{worker_url}/v1/chat/completions", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            c = (msg.get("content") or msg.get("reasoning") or "").strip()
            if c and len(c) > 30:  # filter out empty / too-short
                return c
        except Exception:
            if attempt == RETRIES:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def redistill_row(row: dict, workers: list[str], wid_seed: int) -> tuple[dict | None, int, int]:
    """Returns (new_row | None, turns_kept, turns_dropped). Drop a turn if Qwen
    fails to produce a substantive response; keep row if any turn survives."""
    new_turns = []
    n_dropped = 0
    img_bytes = row["images"][0]["bytes"]
    img_url = to_data_url(img_bytes)
    for j, turn in enumerate(row.get("texts") or []):
        u = (turn.get("user") or "").strip()
        if not u:
            n_dropped += 1
            continue
        worker = workers[(wid_seed + j) % len(workers)]
        # The question may or may not have <image> prefix already; pass the clean text
        q_clean = u.replace("<image>\n", "").strip()
        new_a = call_qwen(worker, img_url, q_clean)
        if new_a is None:
            n_dropped += 1
            continue
        # Single-image multi-turn: place <image>\n on the FIRST turn only so
        # parser image-placeholder count matches the 1 image per row. Subsequent
        # turns get the clean question. (Bug seen in spatialsense_gold v1.)
        user_out = (f"<image>\n{q_clean}" if len(new_turns) == 0 else q_clean)
        new_turns.append({"user": user_out, "assistant": new_a})

    if not new_turns:
        return None, 0, n_dropped
    out = dict(row)
    out["texts"] = new_turns
    return out, len(new_turns), n_dropped


def process_shard(shard_path: Path, out_path: Path, workers: list[str]) -> dict:
    tbl = pq.read_table(str(shard_path))
    # Only keep the FineVision-canonical columns; drop quality-score columns
    keep_cols = [c for c in ("images", "texts", "source") if c in tbl.column_names]
    tbl = tbl.select(keep_cols)
    rows = tbl.to_pylist()
    if MAX_ROWS_PER_SHARD is not None:
        rows = rows[:MAX_ROWS_PER_SHARD]
    n_in = len(rows)
    results: list[dict | None] = [None] * n_in
    turn_stats = {"kept": 0, "dropped": 0}
    base_seed = random.randint(0, 1_000_000)

    def task(i: int, row: dict):
        return i, redistill_row(row, workers, base_seed + i * 13)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = [ex.submit(task, i, r) for i, r in enumerate(rows)]
        done = 0
        for f in as_completed(futs):
            i, (new_row, n_kept, n_dropped) = f.result()
            results[i] = new_row
            turn_stats["kept"] += n_kept
            turn_stats["dropped"] += n_dropped
            done += 1
            if done % SAMPLE_EVERY == 0:
                samples = [r for r in results[:done] if r is not None]
                if samples:
                    s = random.choice(samples)
                    t0 = s["texts"][0]
                    sys.stderr.write(
                        f"  [QC @ {done}/{n_in}, kept-rows={len(samples)}, "
                        f"turns kept={turn_stats['kept']:,} dropped={turn_stats['dropped']:,}] "
                        f"Q={t0['user'][:100]!r} A={t0['assistant'][:200]!r}\n")

    kept_rows = [r for r in results if r is not None]
    if kept_rows:
        out_tbl = pa.Table.from_pylist(kept_rows, schema=tbl.schema)
        tmp = out_path.with_suffix(".parquet.tmp")
        pq.write_table(out_tbl, str(tmp), compression="zstd")
        tmp.rename(out_path)
    return {"in": n_in, "kept": len(kept_rows),
            "turns_kept": turn_stats["kept"], "turns_dropped": turn_stats["dropped"]}


def main():
    workers = load_workers()
    if not workers:
        sys.stderr.write("no workers in /tmp/ready_workers.txt\n")
        sys.exit(1)
    print(f"workers ({len(workers)})", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shards = sorted(SRC_DIR.glob("train-*.parquet"))
    print(f"=== {len(shards)} shards ===", flush=True)

    t0 = time.time()
    total_in = total_kept = total_turns_kept = total_turns_dropped = 0
    for sh in shards:
        out_path = OUT_DIR / sh.name
        if out_path.exists() and out_path.stat().st_size > 0:
            n = pq.read_metadata(out_path).num_rows
            print(f"  [{sh.name}] SKIP ({n:,} rows)", flush=True)
            total_kept += n
            continue
        t_shard = time.time()
        stats = process_shard(sh, out_path, workers)
        dt = time.time() - t_shard
        total_in += stats["in"]
        total_kept += stats["kept"]
        total_turns_kept += stats["turns_kept"]
        total_turns_dropped += stats["turns_dropped"]
        print(f"  [{sh.name}] rows in={stats['in']:>4,} kept={stats['kept']:>4,} "
              f"turns kept={stats['turns_kept']:>5,} dropped={stats['turns_dropped']:>4,} "
              f"in {dt:.0f}s ({stats['in']/dt:.1f} rows/s) | "
              f"running total: rows={total_kept:,} turns={total_turns_kept:,} elapsed={(time.time()-t0)/60:.1f}m",
              flush=True)

    print(f"\n=== TOTAL ===")
    print(f"  rows in:        {total_in:,}")
    print(f"  rows kept:      {total_kept:,}")
    print(f"  turns kept:     {total_turns_kept:,}")
    print(f"  turns dropped:  {total_turns_dropped:,}")
    print(f"  elapsed: {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
