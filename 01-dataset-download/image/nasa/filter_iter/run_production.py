"""Production two-stage pipeline over the full NASA image set.

Design:
- Target ID list: /capstor/.../nasa_cooldown/keep_ids.parquet (~37K rows).
- Image + description source: /capstor/.../web___nasa___images/shards/*.parquet
  (image_bytes inline, 467 shards).
- Scan shards once via pyarrow iter_batches; for each row whose nasa_id is in
  the target set, push a pipeline job into the API thread pool.
- Stage 1 (filter): prompts/v3.txt, thinking OFF.
- Stage 2 (regen KEEP): regen_prompts/balanced.txt, thinking ON.
- Stage 2 (regen DROP): regen_prompts/balanced.txt, thinking OFF, seed replaced.
- Incremental output: write results to Parquet in chunks of N. Resume-safe: on
  restart, already-processed nasa_ids are skipped.
"""
import argparse
import base64
import glob
import io
import json
import os
import random
import re
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import requests
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from clean_caption import clean_caption  # noqa: E402

TARGET_PARQUET = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/nasa_cooldown/keep_ids.parquet")
SHARDS_GLOB = "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/web___nasa___images/shards/*.parquet"
FILTER_PROMPT = (HERE / "prompts" / "v3.txt").read_text()
REGEN_PROMPT = (HERE / "regen_prompts" / "balanced.txt").read_text()
MODEL = "Qwen/Qwen3.6-27B-BZji"
MAX_LONG_SIDE = 1024
EMPTY_SEED_INSTR = "[No seed provided — caption the image from pixels alone. Do not invent identity, names, missions, locations, or dates.]"
THRIFT_LIMIT = 100_000_000  # for parquet metadata

VERDICT_RE = re.compile(r"VERDICT:\s*(KEEP|DROP)", re.IGNORECASE)


def encode_image(image_bytes) -> str | None:
    """Decode + resize + JPEG re-encode + base64. Returns None on any failure."""
    if not image_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Force load to trigger decoding errors early
        img.load()
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            return None
        if max(w, h) > MAX_LONG_SIDE:
            s = MAX_LONG_SIDE / max(w, h)
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def call(endpoint, prompt_text, img_b64, thinking=False, max_tokens=None,
         timeout=300, max_retries=3):
    """POST to the vllm endpoint with bounded retries and exponential backoff."""
    if max_tokens is None:
        max_tokens = 16384 if thinking else 400
    msg = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt_text},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(f"{endpoint}/v1/chat/completions", json=msg,
                              timeout=timeout, proxies={"http": None, "https": None})
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "").strip()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
        except requests.HTTPError as e:
            # 5xx is retryable; 4xx is not
            if 500 <= e.response.status_code < 600:
                last_err = e
            else:
                raise
        except Exception as e:
            last_err = e
        # Backoff before retry
        if attempt < max_retries - 1:
            time.sleep(min(2 ** attempt + random.random(), 10))
    raise RuntimeError(f"call failed after {max_retries} attempts: {last_err}")


def pipeline_one(endpoint, nasa_id, raw_desc, image_bytes,
                 filter_thinking=False, keep_thinking=True, drop_thinking=False):
    try:
        seed = clean_caption(raw_desc or "")
        img_b64 = encode_image(image_bytes)
        if img_b64 is None:
            return {"nasa_id": nasa_id, "seed": seed, "verdict": None,
                    "caption": None, "error": "image_decode_failed"}

        # Stage 1: filter
        f_text = FILTER_PROMPT.format(caption=seed)
        f_out = call(endpoint, f_text, img_b64, thinking=filter_thinking,
                     max_tokens=(8192 if filter_thinking else 128))
        v = VERDICT_RE.search(f_out)
        verdict = v.group(1).upper() if v else "UNKNOWN"

        # Stage 2: regen
        if verdict == "KEEP":
            regen_seed, regen_think = seed, keep_thinking
        else:
            regen_seed, regen_think = EMPTY_SEED_INSTR, drop_thinking
        r_text = REGEN_PROMPT.format(caption=regen_seed)
        caption = call(endpoint, r_text, img_b64, thinking=regen_think)
        # Fallback: if thinking blew the budget, retry without thinking
        if regen_think and not caption.strip():
            caption = call(endpoint, r_text, img_b64, thinking=False)

        return {"nasa_id": nasa_id, "seed": seed, "verdict": verdict,
                "caption": caption, "error": None}
    except Exception as e:
        return {"nasa_id": nasa_id, "seed": clean_caption(raw_desc or ""),
                "verdict": None, "caption": None, "error": str(e)}


class IncrementalParquetWriter:
    """Appends batches of result dicts to a parquet file. Resume-safe via a
    sidecar JSON set of completed nasa_ids."""

    def __init__(self, out_path: Path, flush_every: int = 500):
        self.out_path = out_path
        self.done_path = out_path.with_suffix(".done.json")
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._buffer: list[dict] = []
        self._total_written = 0
        # Load completed set
        if self.done_path.exists():
            self.done = set(json.loads(self.done_path.read_text()))
        else:
            self.done = set()
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()

    def append(self, row: dict):
        with self._lock:
            self._buffer.append(row)
            self.done.add(row["nasa_id"])
            if len(self._buffer) >= self.flush_every:
                self._flush()

    _SCHEMA = {
        "nasa_id": pl.String,
        "seed":    pl.String,
        "verdict": pl.String,
        "caption": pl.String,
        "error":   pl.String,
    }

    def _rows_to_df(self, rows):
        # Explicit schema — avoids polars inferring a Null column from early
        # all-None rows and then failing to coerce later string values.
        norm = [{
            "nasa_id": str(r.get("nasa_id") or ""),
            "seed":    str(r.get("seed") or ""),
            "verdict": r.get("verdict"),
            "caption": r.get("caption"),
            "error":   r.get("error"),
        } for r in rows]
        return pl.DataFrame(norm, schema=self._SCHEMA, strict=False)

    def _flush(self):
        if not self._buffer:
            return
        try:
            df = self._rows_to_df(self._buffer)
        except Exception as e:
            print(f"  [warn] _rows_to_df failed ({e}); coercing one-by-one", flush=True)
            good = []
            for r in self._buffer:
                try:
                    self._rows_to_df([r]); good.append(r)
                except Exception as inner:
                    print(f"  [warn] dropping row {r.get('nasa_id')}: {inner}", flush=True)
            if not good:
                self._buffer.clear(); return
            df = self._rows_to_df(good)

        if self.out_path.exists():
            try:
                existing = pl.read_parquet(self.out_path)
                df = pl.concat([existing, df], how="vertical_relaxed")
            except Exception as e:
                backup = self.out_path.with_suffix(".corrupt.bak")
                self.out_path.rename(backup)
                print(f"  [warn] existing parquet unreadable ({e}); moved to {backup}", flush=True)

        try:
            tmp = self.out_path.with_suffix(".tmp")
            df.write_parquet(tmp, compression="zstd")
            os.replace(tmp, self.out_path)
            self._total_written += len(self._buffer)
            self._buffer.clear()
            tmp_done = self.done_path.with_suffix(".tmp")
            tmp_done.write_text(json.dumps(sorted(self.done)))
            os.replace(tmp_done, self.done_path)
        except Exception as e:
            print(f"  [error] flush write failed ({e}); buffer RETAINED, will retry next flush", flush=True)
            # do NOT clear buffer on write failure; next _flush will retry

    def close(self):
        with self._lock:
            self._flush()

    def total(self):
        return self._total_written + len(self._buffer)


def _scan_shard_polars(shard, target_list):
    """Primary path: polars streaming scan. Returns list of row dicts or None."""
    try:
        df = (pl.scan_parquet(shard)
              .filter(pl.col("nasa_id").is_in(target_list))
              .select(["nasa_id", "description", "image_bytes"])
              .collect(engine="streaming"))
        return list(df.iter_rows(named=True))
    except Exception:
        return None


def _scan_shard_pyarrow(shard, target_set):
    """Fallback: pyarrow iter_batches with per-batch try/except, so a single
    corrupted page only skips that page, not the whole shard."""
    try:
        pf = pq.ParquetFile(shard,
                            thrift_string_size_limit=THRIFT_LIMIT,
                            thrift_container_size_limit=THRIFT_LIMIT)
    except Exception:
        return None
    cols = ["nasa_id", "description", "image_bytes"]
    matched = []
    try:
        batch_iter = pf.iter_batches(batch_size=64, columns=cols)
    except Exception:
        return None
    while True:
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        except Exception:
            # skip this batch, continue to next
            continue
        try:
            nids = batch.column("nasa_id").to_pylist()
        except Exception:
            continue
        for i, nid in enumerate(nids):
            if nid not in target_set:
                continue
            try:
                desc = batch.column("description")[i].as_py()
                img_b = batch.column("image_bytes")[i].as_py()
            except Exception:
                continue
            matched.append({"nasa_id": nid, "description": desc, "image_bytes": img_b})
    return matched


def iter_target_rows(target_ids: set[str], limit: int | None = None):
    """Iterate (nasa_id, description, image_bytes) for rows whose nasa_id is in
    target_ids. Visits each shard once. Primary loader is polars streaming; on
    failure, falls back to pyarrow iter_batches with per-batch recovery."""
    shards = sorted(glob.glob(SHARDS_GLOB))
    target_list = list(target_ids)
    yielded = 0
    stats = {"shards_ok": 0, "shards_polars_failed_fallback_ok": 0,
             "shards_pyarrow_partial": 0, "shards_fully_failed": 0,
             "rows_skipped_missing_bytes": 0, "total_matched_rows": 0}

    for shard_idx, shard in enumerate(shards):
        name = Path(shard).name
        rows = _scan_shard_polars(shard, target_list)
        if rows is None:
            # polars failed — try pyarrow fallback
            rows = _scan_shard_pyarrow(shard, target_ids)
            if rows is None:
                stats["shards_fully_failed"] += 1
                print(f"  [warn] shard {name}: both polars AND pyarrow failed, skipping", flush=True)
                continue
            else:
                stats["shards_polars_failed_fallback_ok"] += 1
                print(f"  [warn] shard {name}: polars failed, recovered {len(rows)} rows via pyarrow", flush=True)
        else:
            stats["shards_ok"] += 1

        for row in rows:
            nid = row["nasa_id"]
            desc = row["description"]
            img = row["image_bytes"]
            if not img:
                stats["rows_skipped_missing_bytes"] += 1
                continue
            yield (nid, desc, img)
            yielded += 1
            stats["total_matched_rows"] += 1
            if limit and yielded >= limit:
                print(f"  [iter] reached limit={limit}; stats={stats}", flush=True)
                return

    print(f"  [iter] scan complete; stats={stats}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://172.28.51.187:8080")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="smoke-test cap on targets processed")
    ap.add_argument("--filter-thinking", action="store_true", help="thinking ON for filter stage (default OFF)")
    ap.add_argument("--no-keep-thinking", action="store_true", help="disable thinking for regen KEEP branch (default ON)")
    ap.add_argument("--drop-thinking", action="store_true", help="enable thinking for regen DROP branch (default OFF)")
    ap.add_argument("--out", default=str(HERE / "production_captions.parquet"))
    ap.add_argument("--flush-every", type=int, default=500)
    args = ap.parse_args()

    target_ids = set(pl.read_parquet(TARGET_PARQUET)["nasa_id"].to_list())
    print(f"target IDs: {len(target_ids)}")

    writer = IncrementalParquetWriter(Path(args.out), flush_every=args.flush_every)
    if writer.done:
        print(f"resume: {len(writer.done)} already processed, skipping those")
        target_ids -= writer.done
        print(f"remaining: {len(target_ids)}")

    if args.limit:
        print(f"smoke-test limit: {args.limit}")

    filter_thinking = args.filter_thinking
    keep_thinking = not args.no_keep_thinking
    drop_thinking = args.drop_thinking
    print(f"thinking flags: filter={filter_thinking} keep={keep_thinking} drop={drop_thinking}")
    print(f"concurrency: {args.concurrency}  output: {args.out}")

    # Graceful shutdown on SIGINT/SIGTERM: flush writer, then exit.
    stop_requested = {"v": False}
    def handle_signal(signum, frame):
        if stop_requested["v"]:
            print("\n[signal] second interrupt — exiting immediately", flush=True)
            os._exit(130)
        stop_requested["v"] = True
        print(f"\n[signal] received {signum}; will flush and exit after in-flight work completes (Ctrl-C again to force)", flush=True)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    t0 = time.time()
    n_done = 0
    n_error = 0
    verdict_counts = {"KEEP": 0, "DROP": 0, "UNKNOWN": 0}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {}
        # Producer: enqueue up to 2x concurrency in-flight
        iterator = iter_target_rows(target_ids, limit=args.limit)
        in_flight = 0
        queue_cap = args.concurrency * 2

        def submit_next():
            nonlocal in_flight
            try:
                nid, desc, img_bytes = next(iterator)
            except StopIteration:
                return False
            fut = ex.submit(pipeline_one, args.endpoint, nid, desc, img_bytes,
                            filter_thinking, keep_thinking, drop_thinking)
            futures[fut] = nid
            in_flight += 1
            return True

        # Fill the queue
        for _ in range(queue_cap):
            if not submit_next():
                break

        while futures:
            for fut in as_completed(list(futures.keys())):
                try:
                    result = fut.result()
                except Exception as e:
                    # Safety net: pipeline_one should already catch, but if a worker
                    # escaped somehow, synthesize an error row using the nasa_id we stored.
                    result = {"nasa_id": futures[fut], "seed": "", "verdict": None,
                              "caption": None, "error": f"worker_exception: {type(e).__name__}: {e}"}
                del futures[fut]
                in_flight -= 1
                try:
                    writer.append(result)
                except Exception as e:
                    print(f"  [warn] writer.append failed for {result.get('nasa_id')}: {e}", flush=True)
                n_done += 1
                verdict_counts[result.get("verdict") or "UNKNOWN"] = verdict_counts.get(result.get("verdict") or "UNKNOWN", 0) + 1
                if result.get("error"):
                    n_error += 1
                if n_done % 100 == 0 or n_done == 1:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 0.001)
                    eta_rem = (len(target_ids) - n_done) / max(rate, 0.001) if not args.limit else (min(args.limit, len(target_ids)) - n_done) / max(rate, 0.001)
                    print(f"  [{n_done}] {elapsed:.0f}s  {rate:.1f} req/s  eta {eta_rem/60:.1f}m  verdicts={verdict_counts}  err={n_error}")
                # Graceful stop: stop pulling new work
                if stop_requested["v"]:
                    break
                # Replenish
                if not args.limit or n_done + in_flight < args.limit:
                    submit_next()
                break  # process one, then continue the outer while
            if stop_requested["v"] and not futures:
                break

    writer.close()
    print(f"\nDONE. {n_done} processed in {(time.time()-t0)/60:.1f} min")
    print(f"verdicts: {verdict_counts}  errors: {n_error}")
    print(f"output:   {args.out}")


if __name__ == "__main__":
    main()
