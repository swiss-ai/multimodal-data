"""Production WAFFLE captioner — streams all parquets, writes caption shards.

Usage:
    python caption_waffle.py --endpoint http://<router-ip>:30000
    # or let it auto-discover from ~/.sml/logs/<latest>/log.out

Resume: skips page_ids already captioned in the output dir. Safe to restart.
"""
import argparse
import asyncio
import base64
import glob
import io
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import httpx
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


# -------- config --------

MODEL = "Qwen/Qwen3.5-397B-A17B-" + os.environ.get("USER", "xyixuan")
WAFFLE_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/tau-vailab___WAFFLE/parquet")
OUTPUT_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/waffle_captions")
MAX_LONGEST_SIDE = 2048
MAX_TOKENS = 2400
TEMPERATURE = 0.3
FLUSH_EVERY = 500           # rows per output shard
PROGRESS_EVERY_SEC = 30


SYSTEM_PROMPT = """You produce captions for a vision-language training dataset. A downstream model will learn from these captions, so every word must be clean training signal: dense visual description, grounded in what is literally visible in the image. Your job is to REPORT what is on the page, not to INTERPRET, narrate, evaluate, or tell its story.

Write ONE description, as long as the image warrants. No headers, no bullets, no preamble ("This image shows...", "The drawing depicts...", "In the foreground..."). No closing summary ("Overall,", "In summary", "A testament to..."). Begin with a concrete description of what occupies the image and continue until every visible element has been described.

HARD RULES

1. DESCRIBE, DO NOT INTERPRET. Report what is drawn, labeled, shown, hatched, dimensioned. Do NOT explain function, history, purpose, intent, or meaning. BAD: "the machinery dictates the primary grinding axis", "a testament to industrial adaptation", "reads like a catalog of engineering", "sits ready for the final stage of processing". GOOD: "four circled numerals labeled 1 through 4 sit along a straight line running east-west, each labeled 'Flour Roller Mills'".

2. NO SPECULATION. Ban these words and their synonyms entirely: "likely", "perhaps", "possibly", "suggesting", "evoking", "reminiscent of", "characteristic of", "indicative of", "as if", "akin to", "hints at", "speaks to", "tells a story", "seems", "appears to". If you are tempted to hedge, you are interpreting — stop and report what you actually see instead.

3. NO EVALUATIVE OR LITERARY LANGUAGE. Ban these unless directly justified by specific visible evidence: "stark", "surgical", "meticulous", "exquisite", "grand", "monumental", "delicate", "intricate", "elegant", "striking", "handsome", "pure", "unified", "masterful", "testament to". "Hand-drawn" is allowed because ink on paper is visible; "crisp linework" is NOT unless you are distinguishing it from visibly blurred linework in the same image.

4. NO CONTENT NOT VISIBLE IN THE IMAGE. Do NOT reference anything outside the frame ("the unseen water wheel", "the adjacent room", "the hidden foundation"). Do NOT import dates, names, histories, or functional narratives that aren't printed on the page. If a date, name, or description appears in the image's text (title block, annotations, stamps), you may quote it exactly.

5. TRANSCRIBE ALL LEGIBLE TEXT. Labels, dimensions, scale indicators, sheet numbers, survey codes, project titles, draftsmen's names, dates, manufacturer marks, patent numbers, location text, key map notes, legend keys — transcribe them verbatim, in quotes. Building names printed in title blocks ARE visible text; include them.

6. TRANSCRIBE DISCRETE ENUMERATED LISTS IN FULL. This rule applies ONLY to bounded, clearly-delimited lists: a numbered legend/key, a window/door/profile schedule, a parts index, a multi-view callout set (PLAN-ON-A-A, PLAN-ON-B-B, ...), a bounded table with labeled rows. When such a list is present, transcribe every row. A 22-item legend yields 22 entries; a 35-row schedule yields 35 rows. Never write "among others" or "and so on" for these.

   This rule does NOT apply to scattered labels distributed across a drawing — tree survey dots, planting schedule points, grid reference numbers, contour elevation tags, PP-codes dispersed around a site plan, or any set of markers that would require listing dozens-to-hundreds of individual numbers. For these, DO NOT attempt exhaustive transcription. Instead describe them as a class with: (a) approximate count or density ("roughly 200 numbered circles", "dozens of elevation tags"), (b) the range or pattern of the numbering you can confidently read ("numbering runs from 1 to approximately 220", "elevations visible from 1535 down to 1480"), (c) 3-5 concrete example numbers you can literally read from the image. Do not generate sequences beyond what you can actually see.

7. NEVER CONFABULATE TO SATISFY A RULE. If you cannot clearly read an entry, omit it. If a list is too dense to transcribe fully, describe it as a class (see rule 6). It is better to be brief and honest than to produce a plausible-sounding but invented sequence. Duplicated entries, implausible sequences, or patterned extrapolations are hallucination — worse than no transcription.

8. ENUMERATE EACH VIEW SEPARATELY. If a sheet contains multiple plans, sections, elevations, or diagrams, describe each as a distinct block: name it ("First Floor Plan", "PLAN-ON-C-C", "Section A-A", "Roof Plan"), then describe what is drawn within it. Do not fold them into a generic composite.

9. MATERIAL INFERENCE IS ALLOWED BUT LIMITED. You may name a material when line weight, hatching, or stippling conventionally denotes it (stippled fill = masonry/stone; diagonal hatch = cut section; solid = wall; dashed = hidden/above). When you do, state the visual cue: "diagonal hatching denotes cut masonry", not just "masonry walls".

10. NEUTRAL VOICE. No "we see", no "the viewer", no "the eye", no first-person. No rhetorical questions. No narrative arc. Just description. Vary sentence openings to avoid starting every sentence with "The plan..." / "The drawing..." / "There is...", but do not introduce voice in the process.

11. PUNCTUATION AND QUOTES. Put every piece of quoted text in the image inside double quotes. Preserve exact capitalization and punctuation from the image.

Output begins with the first word of the description. No preface.
"""
USER_PROMPT = "Caption this image following the rules in the system prompt."


# -------- helpers --------

def _extract_urls_from_log(text: str) -> tuple[str | None, list[str]]:
    """Return (router_url, [worker_url, ...]) from an sml job log."""
    router = None
    workers: list[str] = []
    m = re.search(r"Router URL:\s*(http://\S+)", text)
    if m:
        router = m.group(1)
    m = re.search(r"All worker URLs:\s*(.+)", text)
    if m:
        workers = [u for u in m.group(1).split() if u.startswith("http")]
    return router, workers


def discover_endpoints_from_job(job_id: str) -> list[str]:
    """Read an sml job log, return its worker URLs (preferred) or router URL.

    Worker URLs are preferred over the router because when the router's node
    dies (as happens in partial-boot failures) the router is unreachable even
    though other workers are healthy. A direct list of worker URLs with
    client-side round-robin survives single-worker failures gracefully.
    """
    log = Path.home() / ".sml/logs" / job_id / "log.out"
    if not log.is_file():
        raise RuntimeError(f"sml log does not exist: {log}")
    text = log.read_text()
    router, workers = _extract_urls_from_log(text)
    if workers:
        return workers
    if router:
        return [router]
    raise RuntimeError(
        f"no 'All worker URLs:' or 'Router URL:' line found in {log}. "
        f"Job may still be booting — check its status and retry."
    )


def auto_discover_endpoints() -> list[str]:
    """Find endpoints via newest-mtime numeric job-log directory.

    Prefer ``--job-id`` for deterministic behaviour in production runs.
    """
    logs_dir = Path.home() / ".sml/logs"
    if not logs_dir.is_dir():
        raise RuntimeError(f"sml logs dir does not exist: {logs_dir}")

    candidates = []
    for p in logs_dir.iterdir():
        if not p.is_dir() or not p.name.isdigit():
            continue
        log = p / "log.out"
        if log.is_file():
            candidates.append((log.stat().st_mtime, p))
    if not candidates:
        raise RuntimeError(
            f"no valid sml job-log directories found under {logs_dir}. "
            f"Make sure an sml job has been submitted and has started writing log.out."
        )

    for _, p in sorted(candidates, reverse=True):
        try:
            return discover_endpoints_from_job(p.name)
        except RuntimeError:
            continue

    raise RuntimeError(
        f"scanned {len(candidates)} sml log dirs in {logs_dir}, none contained a "
        f"router or worker URL. Pass --job-id or --endpoint explicitly."
    )


def health_filter(urls: list[str], timeout: float = 5.0) -> list[str]:
    """Probe /health on each URL, return the subset that returns 200.

    Dead workers (failed init, stuck loading, OOM'd) are dropped so the
    client doesn't route traffic to them. Exits with a useful error if
    zero workers are healthy.
    """
    alive: list[str] = []
    dead: list[tuple[str, str]] = []
    for url in urls:
        base = url.rstrip("/").removesuffix("/v1/chat/completions")
        probe = base + "/health"
        try:
            r = httpx.get(probe, timeout=timeout)
            if r.status_code == 200:
                alive.append(base)
            else:
                dead.append((base, f"HTTP {r.status_code}"))
        except Exception as e:
            dead.append((base, f"{type(e).__name__}: {str(e)[:80]}"))

    if not alive:
        msg = "\n  ".join(f"{u}  [{why}]" for u, why in dead)
        raise RuntimeError(
            f"No healthy endpoints.\n  {msg}\n"
            f"Check the sml job status with `squeue -u $USER` and the log in "
            f"~/.sml/logs/<job-id>/log.out."
        )
    if dead:
        print(f"dropping {len(dead)} dead endpoints:", flush=True)
        for u, why in dead:
            print(f"  {u}  [{why}]", flush=True)
    return alive


def resize_jpeg(raw: bytes, target: int = MAX_LONGEST_SIDE) -> bytes:
    im = Image.open(io.BytesIO(raw))
    w, h = im.size
    if max(w, h) <= target:
        return raw
    s = target / max(w, h)
    im = im.convert("RGB").resize((int(w*s), int(h*s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def load_done_page_ids(out_dir: Path) -> set[str]:
    """Scan existing output parquets for already-captioned page_ids."""
    done = set()
    for p in sorted(out_dir.glob("captions_*.parquet")):
        try:
            df = pl.read_parquet(p, columns=["page_id"])
            done.update(df["page_id"].to_list())
        except Exception as e:
            print(f"WARN: could not read {p}: {e}", file=sys.stderr)
    return done


def stream_rows(done: set[str]):
    """Yield (bucket, row) tuples from permissive parquets, skipping done.

    Streams via pyarrow row-group batches so we never hold a full shard's
    `image_bytes` column in RAM. SA (share-alike) bucket is excluded — we
    only caption the commercial-OK subset.
    """
    for bucket in ("permissive",):
        shards = sorted((WAFFLE_ROOT / bucket).glob("train-*.parquet"))
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=64):
                df = pl.from_arrow(batch)
                for row in df.iter_rows(named=True):
                    if row["page_id"] in done:
                        continue
                    yield bucket, row


class ShardWriter:
    """Append rows to numbered output shards, flushing every FLUSH_EVERY."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.out_dir.glob("captions_*.parquet"))
        self.idx = int(re.search(r"(\d+)", existing[-1].name).group(1)) + 1 if existing else 0
        self.buffer: list[dict] = []

    def add(self, row: dict):
        self.buffer.append(row)
        if len(self.buffer) >= FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        out = self.out_dir / f"captions_{self.idx:05d}.parquet"
        table = pl.DataFrame(self.buffer).to_arrow()
        pq.write_table(table, out, compression="snappy")
        print(f"  flushed {len(self.buffer)} rows -> {out.name}", flush=True)
        self.buffer.clear()
        self.idx += 1


# -------- async inference --------

def make_row(page_id, bucket, license_id, caption="", latency_s=0.0, error=""):
    """Build an output row with the full, stable schema. Used for both success
    and failure paths so every output shard has identical columns."""
    return {
        "page_id": page_id,
        "bucket": bucket,
        "caption": caption,
        "latency_s": round(float(latency_s), 2),
        "word_count": len(caption.split()) if caption else 0,
        "license_id": license_id,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "image_longest_side": MAX_LONGEST_SIDE,
        "error": error,
    }


class EndpointPool:
    """Round-robin pool of healthy endpoints, atomic across asyncio tasks.

    Survives single-endpoint failures: retries within a request pick a
    different endpoint, so a flaky or degraded worker shifts traffic to
    healthy peers without taking down the whole run.
    """

    def __init__(self, urls: list[str]):
        if not urls:
            raise ValueError("EndpointPool requires at least one URL")
        # Normalise to base (no /v1/...) — we append the path at dispatch time
        # so "http://ip:port", "http://ip:port/", "http://ip:port/v1/chat/completions"
        # all work as input.
        normalized: list[str] = []
        for u in urls:
            base = u.rstrip("/")
            if base.endswith("/v1/chat/completions"):
                base = base[: -len("/v1/chat/completions")]
            normalized.append(base)
        self._urls = normalized
        self._i = 0
        self._lock = asyncio.Lock()

    def __len__(self) -> int:
        return len(self._urls)

    @property
    def urls(self) -> list[str]:
        return list(self._urls)

    async def next_chat_url(self) -> str:
        async with self._lock:
            url = self._urls[self._i % len(self._urls)]
            self._i += 1
        return f"{url}/v1/chat/completions"


async def caption_one(client, sem, pool, bucket, row, metrics):
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
        "temperature": TEMPERATURE,
        "chat_template_kwargs": {"enable_thinking": False},
    }
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
                metrics["total_tokens"] += len(reply.split()) * 1.3
                return make_row(page_id, bucket, license_id, caption=reply, latency_s=dt)
            except Exception as e:
                last_error = f"{type(e).__name__}: {str(e)[:200]}"
                if attempt == 2:
                    metrics["fail"] += 1
                    return make_row(page_id, bucket, license_id, error=last_error)
                await asyncio.sleep(5 * (2 ** attempt))


# -------- main --------

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=None,
                    help="Single http://ip:port explicit override; skips log discovery. "
                         "Mutually exclusive with --endpoints.")
    ap.add_argument("--endpoints", default=None,
                    help="Comma-separated list of worker URLs (http://a:p,http://b:p,...). "
                         "Client round-robins across them so a dead worker doesn't take "
                         "down the run. Skips log discovery.")
    ap.add_argument("--job-id", default=None,
                    help="sml SLURM job id of the model-serving job; reads "
                         "~/.sml/logs/<job-id>/log.out to extract worker URLs "
                         "and health-filters them before use.")
    ap.add_argument("--no-health-filter", action="store_true",
                    help="Skip the /health preflight; use all discovered endpoints. "
                         "Default is to probe each URL and drop unresponsive ones.")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    ap.add_argument("--max-rows", type=int, default=None, help="cap for testing")
    args = ap.parse_args()

    # Resolve the endpoint list. Priority: --endpoints > --endpoint > --job-id > auto.
    # Workers preferred over router because the router's node can be one of the
    # failed-boot ranks, making a single-URL router deployment fragile at scale.
    if args.endpoints:
        candidates = [u.strip() for u in args.endpoints.split(",") if u.strip()]
        source = "--endpoints"
    elif args.endpoint:
        candidates = [args.endpoint]
        source = "--endpoint"
    elif args.job_id:
        candidates = discover_endpoints_from_job(args.job_id)
        source = f"--job-id {args.job_id}"
    else:
        candidates = auto_discover_endpoints()
        source = "newest-log heuristic"
    print(f"candidate endpoints ({source}): {len(candidates)}", flush=True)

    if args.no_health_filter:
        healthy = candidates
        print("  (health filter disabled)", flush=True)
    else:
        healthy = health_filter(candidates)
    pool = EndpointPool(healthy)
    print(f"healthy endpoints in pool: {len(pool)}", flush=True)
    for url in pool.urls:
        print(f"  {url}", flush=True)

    out_dir = Path(args.output_dir)
    done = load_done_page_ids(out_dir)
    print(f"already captioned: {len(done):,}", flush=True)

    writer = ShardWriter(out_dir)
    sem = asyncio.Semaphore(args.concurrency)
    metrics = {"ok": 0, "fail": 0, "total_tokens": 0.0}

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    async def progress_ticker(t_start):
        while not stop.is_set():
            await asyncio.sleep(PROGRESS_EVERY_SEC)
            dt = time.time() - t_start
            rate = metrics["ok"] / dt if dt > 0 else 0
            tok_s = metrics["total_tokens"] / dt if dt > 0 else 0
            print(f"[progress] ok={metrics['ok']:,}  fail={metrics['fail']}  "
                  f"elapsed={dt/60:.1f}m  rate={rate*60:.1f}/min  ~{tok_s:.0f} tok/s",
                  flush=True)

    t_start = time.time()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency*2)) as client:
        tasker = asyncio.create_task(progress_ticker(t_start))
        in_flight: set[asyncio.Task] = set()
        seen = 0
        try:
            for bucket, row in stream_rows(done):
                if args.max_rows and seen >= args.max_rows:
                    break
                if stop.is_set():
                    print("received stop signal — draining in-flight", flush=True)
                    break
                t = asyncio.create_task(caption_one(client, sem, pool, bucket, row, metrics))
                in_flight.add(t)
                t.add_done_callback(in_flight.discard)
                seen += 1
                # periodic drain so we don't accumulate unbounded tasks in memory
                if len(in_flight) >= args.concurrency * 4:
                    done_tasks, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    for t in done_tasks:
                        result = t.result()
                        if result:
                            writer.add(result)
            # drain remainder
            while in_flight:
                done_tasks, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for t in done_tasks:
                    result = t.result()
                    if result:
                        writer.add(result)
        finally:
            writer.flush()
            tasker.cancel()

    dt = time.time() - t_start
    print(f"\nFINAL: ok={metrics['ok']:,} fail={metrics['fail']} "
          f"elapsed={dt/60:.1f} min  ({metrics['ok']/dt*60:.1f}/min avg)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
