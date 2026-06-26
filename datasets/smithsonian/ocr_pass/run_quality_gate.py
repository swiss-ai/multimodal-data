"""Smithsonian OCR-worthiness quality gate against 72 Qwen3.6-27B vLLM endpoints.

Image-only judgement: for each image, decides whether full OCR transcription
is worth running for a language-modeling training corpus. Drops pure stamps
(country+denomination only), decorative objects with no text, illegible
blurry text, and plain specimen tags. Keeps articles, books, dense labels,
handwritten letters, captions, posters.

Cheaper than the full OCR pass: image stays the same but the answer is
~30 tokens instead of up to 4096, so each request is ~3× faster.

Output schema (per source shard, joined to cleaned5 by `id`):
    id, gate_elapsed_s, gate_decision (KEEP/DROP/UNKNOWN),
    gate_reason, gate_raw, gate_endpoint
"""
import argparse
import base64
import io
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image

MAX_IMG_DIM = 1280  # smaller than OCR pass (1792) — gate doesn't need fine text detail
MODEL = "Qwen/Qwen3.6-27B-xyixuan"

PROMPT = (
    "You are a curator deciding which museum-archive images deserve full OCR "
    "transcription for a language-modeling training corpus.\n\n"
    "KEEP if the image contains substantial readable text — newspaper or book "
    "pages, posters with body copy, dense printed labels, handwritten letters or "
    "annotations, captions with full sentences, signs with multiple lines.\n\n"
    "DROP if there is no useful text to transcribe. Common drop cases:\n"
    "  - Postal stamps showing only country + denomination + date (trivial, "
    "repetitive across stamps)\n"
    "  - Decorative objects with just a name or logo\n"
    "  - Photographs of artifacts (sculpture, pottery, machinery) with no text\n"
    "  - Specimen tags with only a Latin name and catalog number\n"
    "  - Images where text is too small/blurry/illegible to OCR confidently\n"
    "  - Blank pages, scans of textile/fabric, abstract art\n\n"
    "Reply with EXACTLY two lines:\n"
    "Decision: KEEP\n"
    "Reason: <one short clause, ≤8 words>\n"
    "or\n"
    "Decision: DROP\n"
    "Reason: <one short clause, ≤8 words>\n\n"
    "Use no other text. No preamble, no markdown, no explanation."
)

DEC_RE = re.compile(r"Decision:\s*(KEEP|DROP)", re.IGNORECASE)
REA_RE = re.compile(r"Reason:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def parse_response(txt: str) -> tuple[str, str]:
    """Extract (decision, reason) from model output. Returns (UNKNOWN, raw) on failure."""
    if not txt:
        return "UNKNOWN", ""
    dm = DEC_RE.search(txt)
    rm = REA_RE.search(txt)
    if dm:
        decision = dm.group(1).upper()
        reason = rm.group(1).strip() if rm else ""
        # Trim runaway reasons to 100 chars
        return decision, reason[:100]
    return "UNKNOWN", txt[:200]


class EndpointPool:
    """Round-robin endpoint dispenser. Drops endpoints that fail repeatedly."""
    def __init__(self, endpoints: list[str]):
        self.endpoints = list(endpoints)
        self.failures = Counter()
        self.lock = Lock()
        self.idx = 0

    def get(self) -> str:
        with self.lock:
            if not self.endpoints:
                raise RuntimeError("no healthy endpoints left")
            ep = self.endpoints[self.idx % len(self.endpoints)]
            self.idx += 1
            return ep

    def report_failure(self, ep: str):
        with self.lock:
            self.failures[ep] += 1
            if self.failures[ep] >= 3 and ep in self.endpoints:
                print(f"  [pool] dropping unhealthy endpoint {ep}")
                self.endpoints.remove(ep)


def discover_endpoints(endpoints_dir: Path, expected: int = 72, timeout: int = 600) -> list[str]:
    deadline = time.time() + timeout
    eps = []
    while time.time() < deadline:
        eps = sorted(set(p.read_text().strip() for p in endpoints_dir.glob("*.endpoint")))
        if len(eps) >= expected:
            break
        print(f"  [{time.strftime('%H:%M:%S')}] waiting for endpoints: {len(eps)}/{expected}")
        time.sleep(15)
    print(f"  found {len(eps)} endpoint(s); health-checking...")
    healthy = []
    for ep in eps:
        try:
            r = requests.get(f"http://{ep}/v1/models", timeout=5)
            if r.status_code == 200:
                healthy.append(ep)
        except Exception:
            pass
    print(f"  {len(healthy)}/{len(eps)} endpoints healthy")
    return healthy


def maybe_resize(img_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    if max(img.size) <= MAX_IMG_DIM:
        return img_bytes
    img.thumbnail((MAX_IMG_DIM, MAX_IMG_DIM), Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def send(ep: str, img_bytes: bytes) -> tuple[float, str]:
    img_bytes = maybe_resize(img_bytes)
    img_b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "max_tokens": 128,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(f"http://{ep}/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return elapsed, msg.get("content") or msg.get("reasoning_content") or ""


def process(pool: EndpointPool, sid: str, img_bytes: bytes) -> dict:
    last_err = None
    for attempt in range(3):
        ep = pool.get()
        try:
            elapsed, txt = send(ep, img_bytes)
            decision, reason = parse_response(txt)
            return {
                "id": sid,
                "gate_elapsed_s": round(elapsed, 2),
                "gate_decision": decision,
                "gate_reason": reason,
                "gate_raw": txt[:500],
                "gate_endpoint": ep,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 0:
                print(f"  [err] {sid} attempt={attempt} ep={ep}: {last_err[:200]}", flush=True)
            pool.report_failure(ep)
    return {
        "id": sid, "gate_elapsed_s": -1.0, "gate_decision": "ERROR",
        "gate_reason": str(last_err)[:100], "gate_raw": "",
        "gate_endpoint": "",
    }


def run_shard(shard: Path, pool: EndpointPool, out_dir: Path, parallelism: int, keep_ids: set | None = None):
    out_path = out_dir / shard.name
    if out_path.exists():
        print(f"[skip] {shard.name} (already exists)")
        return
    t0 = time.time()
    table = pq.read_table(shard, columns=["id", "image"])
    rows = table.to_pylist()
    n_total = len(rows)
    if keep_ids is not None:
        rows = [r for r in rows if r["id"] in keep_ids]
    n = len(rows)
    if n == 0:
        print(f"[{shard.name}] 0/{n_total} rows after filter — writing empty parquet")
        pq.write_table(pa.Table.from_pylist([], schema=pa.schema([
            pa.field("id", pa.string()), pa.field("gate_elapsed_s", pa.float64()),
            pa.field("gate_decision", pa.string()), pa.field("gate_reason", pa.string()),
            pa.field("gate_raw", pa.string()), pa.field("gate_endpoint", pa.string()),
        ])), out_path)
        return
    print(f"[{shard.name}] starting on {n}/{n_total} rows (filtered), parallelism={parallelism}")
    results: list[dict] = [None] * n  # type: ignore
    with ThreadPoolExecutor(max_workers=parallelism) as ex:
        fut2idx = {ex.submit(process, pool, r["id"], r["image"]): i for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(fut2idx):
            idx = fut2idx[fut]
            results[idx] = fut.result()
            done += 1
            if done % 200 == 0 or done == n:
                elapsed = time.time() - t0
                rps = done / elapsed if elapsed > 0 else 0
                print(f"  [{shard.name}] {done}/{n} ({rps:.1f} req/s)")
    rs_table = pa.Table.from_pylist(results)
    pq.write_table(rs_table, out_path)
    elapsed = time.time() - t0
    # Quick decision tally
    dec = Counter(r["gate_decision"] for r in results)
    print(f"[{shard.name}] done in {elapsed:.0f}s ({n/elapsed:.1f} req/s) — {dict(dec)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints-dir", required=True)
    ap.add_argument("--input-dir", default="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/smithsonian/smithsonian_cleaned5")
    ap.add_argument("--output-dir", default="/capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_gate")
    ap.add_argument("--parallelism", type=int, default=200, help="concurrent in-flight requests across all 72 endpoints")
    ap.add_argument("--expected-endpoints", type=int, default=72)
    ap.add_argument("--shards", nargs="*", help="subset of shard names; default: all")
    ap.add_argument("--filter-ids-parquet", help="path to parquet with `id` column; only those ids are gated")
    args = ap.parse_args()

    keep_ids = None
    if args.filter_ids_parquet:
        keep_ids = set(pq.read_table(args.filter_ids_parquet, columns=["id"])["id"].to_pylist())
        print(f"loaded {len(keep_ids):,} ids to gate from {args.filter_ids_parquet}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoints = discover_endpoints(Path(args.endpoints_dir), expected=args.expected_endpoints, timeout=300)
    print(f"\nready with {len(endpoints)} endpoints; starting at {time.strftime('%H:%M:%S')}")
    pool = EndpointPool(endpoints)

    shards = sorted(Path(args.input_dir).glob("*.parquet"))
    if args.shards:
        shards = [s for s in shards if s.name in args.shards]
    print(f"will process {len(shards)} shards")

    t0 = time.time()
    total_rows = sum(pq.read_metadata(s).num_rows for s in shards)
    for shard in shards:
        run_shard(shard, pool, out_dir, args.parallelism, keep_ids=keep_ids)

    wall = time.time() - t0
    print(f"\n=== ALL DONE: {total_rows} images in {wall:.0f}s ({total_rows/wall:.1f} req/s aggregate) ===")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
