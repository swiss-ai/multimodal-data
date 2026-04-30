"""Full Smithsonian cleaned5 OCR pass against 20 Qwen3.6-27B vLLM instances.

No router — client distributes work round-robin to all healthy endpoints.
Writes per-source-shard parquet output with id/qwen_chars/tag_counts/transcript columns.
"""
import argparse
import base64
import io
import json
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

MAX_IMG_DIM = 1792  # cap longest side; ~3000 prompt tokens leaves 13K for context+output within max-model-len=16384

MODEL = "Qwen/Qwen3.6-27B-xyixuan"

PROMPT = (
    "You are an OCR assistant for museum archive images. Transcribe ALL text "
    "visible in this image, grouped by region. Pay special attention to "
    "handwriting (cursive, print, or annotation) — read it carefully and "
    "transcribe exactly as written, even if irregular.\n\n"
    "For each distinct text region, output:\n\n"
    "[<type>] On the <brief surface description>:\n"
    "<exact transcription, preserving line breaks and column layout>\n\n"
    "Use one of these type tags (lowercase, in square brackets):\n"
    "  [handwriting]      — cursive or hand-printed text, signatures, annotations\n"
    "  [printed_label]    — printed text on labels, packaging, signs, plaques\n"
    "  [printed_article]  — body text from books, newspapers, posters, flyers, documents\n"
    "  [printed_caption]  — short captions on photographs, museum cards, illustrations\n"
    "  [stamp]            — postal stamps, date stamps, ink stamps, seals\n"
    "  [engraving]        — engraved/etched/embossed text on metal, stone, wood\n"
    "  [digital_display]  — text on a screen or digital readout\n"
    "  [other]            — anything that doesn't fit above\n\n"
    "List regions in natural reading order (top-to-bottom, left-to-right). "
    "Transcribe exactly as written — do not correct spelling, do not paraphrase, "
    "do not repeat the same line over and over (each text element appears once). "
    "Mark illegible portions as [illegible]. "
    "If no text is visible anywhere, respond with exactly: NO_TEXT"
)

TYPE_TAGS = {"handwriting", "printed_label", "printed_article", "printed_caption",
             "stamp", "engraving", "digital_display", "other"}
TAG_RE = re.compile(r"\[([a-z_]+)\]")


def parse_tags(txt: str) -> Counter:
    c = Counter()
    for tag in TAG_RE.findall(txt):
        if tag in TYPE_TAGS:
            c[tag] += 1
    return c


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


def discover_endpoints(endpoints_dir: Path, expected: int = 20, timeout: int = 600) -> list[str]:
    """Wait for endpoint files to appear, then health-check each one."""
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
                print(f"    ✓ {ep}")
            else:
                print(f"    ✗ {ep} (status {r.status_code})")
        except Exception as e:
            print(f"    ✗ {ep} ({type(e).__name__})")
    return healthy


def wait_for_serving(endpoints: list[str], timeout: int = 600):
    """Block until all endpoints answer /v1/models. Drop ones that don't."""
    deadline = time.time() + timeout
    pending = set(endpoints)
    while pending and time.time() < deadline:
        ready = []
        for ep in pending:
            try:
                if requests.get(f"http://{ep}/v1/models", timeout=3).status_code == 200:
                    ready.append(ep)
            except Exception:
                pass
        for ep in ready:
            pending.discard(ep)
        if pending:
            print(f"  [{time.strftime('%H:%M:%S')}] {len(endpoints)-len(pending)}/{len(endpoints)} ready, waiting on {len(pending)}")
            time.sleep(20)
    if pending:
        print(f"  WARN: {len(pending)} endpoints never came up: {pending}")
        for ep in pending:
            endpoints.remove(ep)
    return endpoints


def maybe_resize(img_bytes: bytes) -> bytes:
    """Downscale to MAX_IMG_DIM longest side if larger; otherwise return unchanged."""
    img = Image.open(io.BytesIO(img_bytes))
    if max(img.size) <= MAX_IMG_DIM:
        return img_bytes
    img.thumbnail((MAX_IMG_DIM, MAX_IMG_DIM), Image.Resampling.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
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
        "max_tokens": 1024,
        "temperature": 0.0,
        "repetition_penalty": 1.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(f"http://{ep}/v1/chat/completions", json=payload, timeout=600)
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
            tags = parse_tags(txt) if txt.strip() != "NO_TEXT" else Counter()
            return {
                "id": sid,
                "ocr_elapsed_s": round(elapsed, 2),
                "ocr_chars": len(txt),
                "ocr_no_text": txt.strip() == "NO_TEXT",
                "ocr_tags_json": json.dumps(dict(tags)),
                "ocr_text": txt,
                "ocr_endpoint": ep,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"  [err] {sid} attempt={attempt} ep={ep}: {last_err[:200]}", flush=True)
            pool.report_failure(ep)
    return {
        "id": sid, "ocr_elapsed_s": -1.0, "ocr_chars": 0,
        "ocr_no_text": False, "ocr_tags_json": "{}",
        "ocr_text": f"[ERROR] {last_err}", "ocr_endpoint": "",
    }


def run_shard(shard: Path, pool: EndpointPool, out_dir: Path, parallelism: int):
    """OCR every row in a parquet shard, write back as parquet with new columns."""
    out_path = out_dir / shard.name
    if out_path.exists():
        print(f"[skip] {shard.name} (already exists)")
        return
    t0 = time.time()
    table = pq.read_table(shard, columns=["id", "image"])
    n = table.num_rows
    rows = table.to_pylist()
    print(f"[{shard.name}] starting on {n} rows, parallelism={parallelism}")
    results: list[dict] = [None] * n  # type: ignore
    with ThreadPoolExecutor(max_workers=parallelism) as ex:
        fut2idx = {ex.submit(process, pool, r["id"], r["image"]): i for i, r in enumerate(rows)}
        done = 0
        for fut in as_completed(fut2idx):
            idx = fut2idx[fut]
            results[idx] = fut.result()
            done += 1
            if done % 100 == 0 or done == n:
                elapsed = time.time() - t0
                rps = done / elapsed if elapsed > 0 else 0
                print(f"  [{shard.name}] {done}/{n} ({rps:.1f} req/s)")
    rs_table = pa.Table.from_pylist(results)
    pq.write_table(rs_table, out_path)
    elapsed = time.time() - t0
    print(f"[{shard.name}] done in {elapsed:.0f}s ({n/elapsed:.1f} req/s aggregate)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints-dir", required=True)
    ap.add_argument("--input-dir", default="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/smithsonian/smithsonian_cleaned5")
    ap.add_argument("--output-dir", default="/capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_full")
    ap.add_argument("--parallelism", type=int, default=80, help="concurrent in-flight requests across all endpoints")
    ap.add_argument("--shards", nargs="*", help="subset of shard names to process; default: all")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoints = discover_endpoints(Path(args.endpoints_dir), expected=20, timeout=900)
    endpoints = wait_for_serving(endpoints, timeout=900)
    print(f"\nready with {len(endpoints)} endpoints; starting at {time.strftime('%H:%M:%S')}")
    pool = EndpointPool(endpoints)

    shards = sorted(Path(args.input_dir).glob("*.parquet"))
    if args.shards:
        shards = [s for s in shards if s.name in args.shards]
    print(f"will process {len(shards)} shards")

    t0 = time.time()
    total_rows = 0
    for shard in shards:
        n = pq.read_metadata(shard).num_rows
        total_rows += n
        run_shard(shard, pool, out_dir, args.parallelism)

    wall = time.time() - t0
    print(f"\n=== ALL DONE: {total_rows} images in {wall:.0f}s ({total_rows/wall:.1f} req/s aggregate) ===")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
