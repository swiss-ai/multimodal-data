"""Re-caption the quarantined WAFFLE rows with different sampling.

The production captioner (``caption_waffle.py``) uses temperature=0.3. About
~2.7% of rows fall into a degenerate local attractor (token loop, accretion
loop, quoted-number list). Re-running those specific rows with a different
sampling trajectory usually produces a clean caption because the loop
attractor is run-specific, not image-specific.

Pipeline:
  1. Load ``waffle_captions_quarantine.parquet`` to get the page_ids to retry.
  2. Pull image bytes for those page_ids from the WAFFLE source parquets.
  3. Post each to the live model pool with temperature=0.0 (greedy, avoids
     the sampling-noise attractors) + a fresh seed.
  4. Run the same Tier-1 degen filter on the new captions to see which are
     now clean.
  5. Write ``waffle_captions_regenerated.parquet`` (all retries) and
     ``waffle_captions_still_quarantined.parquet`` (those that stayed bad).
  6. Exit — a follow-up merge step folds the newly-clean rows back into the
     fat training shards.

Usage:
    python regenerate_quarantined.py --job-id 1873962
    # optional:
    python regenerate_quarantined.py --job-id 1873962 --temperature 0.7 \
        --seed 42 --concurrency 32
"""

import argparse
import asyncio
import base64
import io
import re
import time
from pathlib import Path

import httpx
import polars as pl
from PIL import Image

# Reuse the endpoint discovery / health-filter / pool machinery from the
# production captioner — same building blocks, different input stream.
from caption_waffle import (
    EndpointPool,
    SYSTEM_PROMPT,
    USER_PROMPT,
    MODEL,
    MAX_TOKENS,
    MAX_LONGEST_SIDE,
    make_row,
    resize_jpeg,
    health_filter,
    discover_endpoints_from_job,
)
from merge_captions import degen_reason


DEFAULT_CAPTIONS_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/waffle_captions")
DEFAULT_WAFFLE_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/tau-vailab___WAFFLE/parquet/permissive")


async def caption_one(client, sem, pool, bucket, row, metrics, temperature: float,
                      seed: int | None):
    page_id = row["page_id"]
    license_id = row.get("license_id", "")
    try:
        img_b64 = base64.b64encode(resize_jpeg(row["image_bytes"])).decode()
    except Exception as e:
        metrics["fail"] += 1
        return make_row(page_id, bucket, license_id,
                        error=f"resize: {type(e).__name__}: {str(e)[:200]}")
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        body["seed"] = seed
    last_error = ""
    for attempt in range(3):
        endpoint = await pool.next_chat_url()
        async with sem:
            t0 = time.time()
            try:
                r = await client.post(endpoint, json=body, timeout=2400.0)
                r.raise_for_status()
                reply = r.json()["choices"][0]["message"]["content"]
                dt = time.time() - t0
                metrics["ok"] += 1
                return make_row(page_id, bucket, license_id, caption=reply, latency_s=dt)
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt == 2:
                    metrics["fail"] += 1
                    return make_row(page_id, bucket, license_id, error=last_error)
                await asyncio.sleep(5 * (2 ** attempt))


def load_image_bytes_for(page_ids: set[str], waffle_dir: Path) -> dict[str, dict]:
    """Return page_id -> {image_bytes, license_id, bucket} by scanning WAFFLE parquets."""
    import pyarrow.parquet as pq
    out: dict[str, dict] = {}
    remaining = set(page_ids)
    shards = sorted(waffle_dir.glob("train-*.parquet"))
    for shard in shards:
        if not remaining:
            break
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(
            batch_size=512,
            columns=["page_id", "image_bytes", "license_id"],
        ):
            df = pl.from_arrow(batch)
            for row in df.iter_rows(named=True):
                pid = row["page_id"]
                if pid in remaining:
                    out[pid] = {
                        "page_id": pid,
                        "image_bytes": row["image_bytes"],
                        "license_id": row.get("license_id", ""),
                    }
                    remaining.discard(pid)
                    if not remaining:
                        break
            if not remaining:
                break
    if remaining:
        print(f"WARN: {len(remaining)} page_ids not found in WAFFLE source "
              f"(possibly from sa/ bucket — not regenerating)", flush=True)
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--captions-dir", type=Path, default=DEFAULT_CAPTIONS_DIR)
    ap.add_argument("--waffle-dir", type=Path, default=DEFAULT_WAFFLE_DIR)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    q_path = args.captions_dir / "waffle_captions_quarantine.parquet"
    if not q_path.is_file():
        raise FileNotFoundError(f"expected quarantine parquet at {q_path}; "
                                "run merge_captions.py first")
    quarantine = pl.read_parquet(q_path)
    page_ids = set(quarantine["page_id"].to_list())
    print(f"loaded {len(page_ids)} quarantined page_ids from {q_path}", flush=True)

    rows = load_image_bytes_for(page_ids, args.waffle_dir)
    print(f"loaded {len(rows)} images from WAFFLE source", flush=True)

    candidates = discover_endpoints_from_job(args.job_id)
    print(f"candidate endpoints: {len(candidates)}", flush=True)
    healthy = health_filter(candidates)
    pool = EndpointPool(healthy)
    print(f"healthy endpoints in pool: {len(pool)}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    metrics = {"ok": 0, "fail": 0}

    t_start = time.time()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency * 2)) as client:
        tasks = [
            caption_one(client, sem, pool, "permissive", row, metrics,
                        temperature=args.temperature, seed=args.seed)
            for row in rows.values()
        ]
        print(f"firing {len(tasks)} re-captions (temperature={args.temperature}, seed={args.seed}, "
              f"concurrency={args.concurrency})", flush=True)
        results = await asyncio.gather(*tasks)

    dt = time.time() - t_start
    print(f"\nFINAL: ok={metrics['ok']} fail={metrics['fail']} elapsed={dt/60:.1f} min "
          f"({metrics['ok']/dt*60:.1f}/min avg)", flush=True)

    # Re-run Tier-1 degen filter on the new captions
    reasons = [degen_reason(r["caption"]) or ("error:" + r["error"] if r["error"] else "") for r in results]
    new_df = pl.DataFrame(results).with_columns(
        pl.Series("quarantine_reason", reasons)
    )
    clean_new = new_df.filter(pl.col("quarantine_reason") == "")
    still_bad = new_df.filter(pl.col("quarantine_reason") != "")
    print(f"\npost-regeneration Tier-1:", flush=True)
    print(f"  cleaned: {len(clean_new)}/{len(new_df)} "
          f"({len(clean_new)/len(new_df)*100:.1f}%)", flush=True)
    print(f"  still degenerate: {len(still_bad)} "
          f"({len(still_bad)/len(new_df)*100:.1f}%)", flush=True)

    out_all = args.captions_dir / "waffle_captions_regenerated.parquet"
    out_still = args.captions_dir / "waffle_captions_still_quarantined.parquet"
    new_df.write_parquet(out_all, compression="zstd")
    still_bad.write_parquet(out_still, compression="zstd")
    print(f"\nwrote {out_all}  ({len(new_df)} rows)", flush=True)
    print(f"wrote {out_still}  ({len(still_bad)} rows)", flush=True)
    print("\nnext: run `python merge_regenerated.py` to fold cleaned rows into "
          "the clean fat shards.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
