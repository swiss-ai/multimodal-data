"""Smithsonian OCR verifier against 72 Qwen3.6-27B vLLM endpoints.

For each (image, prior_ocr_text) pair, asks Qwen to grade the OCR transcription
on a 1-5 scale and flag the dominant error type. Used as stage 3 of the
"perfect OCR dataset" pipeline:

    cleaned5 → OCR (stage 0) → platinum filter (stage 1) → image gate (stage 2)
        → THIS verifier (stage 3) → post-filters (stage 4) → final assembly

Output schema (per source shard, joined to gate output by `id`):
    id, ver_elapsed_s, ver_grade (1-5 or 0=parse-fail), ver_issues, ver_raw, ver_endpoint
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

MAX_IMG_DIM = 1792  # match OCR pass — verifier must see what OCR saw
MODEL = "Qwen/Qwen3.6-27B-xyixuan"

PROMPT_TMPL = (
    "Below is OCR output from a museum-archive image. Look at the image and "
    "judge whether the OCR transcription is accurate.\n\n"
    "OCR text:\n\"\"\"\n{ocr_text}\n\"\"\"\n\n"
    "Reply with EXACTLY two lines:\n"
    "Grade: <1-5>\n"
    "Issues: <one short clause, ≤10 words; 'none' if grade ≥ 4>\n\n"
    "Grading rubric:\n"
    "  5 = perfect — every visible word transcribed exactly\n"
    "  4 = good — only minor errors (1-2 misspellings, missed punctuation)\n"
    "  3 = mixed — several errors but core content right\n"
    "  2 = poor — many wrong words, missing major chunks, or misordered\n"
    "  1 = severely wrong — hallucinations, made-up content, or unrelated text\n\n"
    "Use no other text. No preamble, no markdown, no explanation."
)

GRADE_RE  = re.compile(r"Grade:\s*([1-5])", re.IGNORECASE)
ISSUES_RE = re.compile(r"Issues:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def parse_response(txt: str) -> tuple[int, str]:
    if not txt:
        return 0, ""
    gm = GRADE_RE.search(txt)
    im = ISSUES_RE.search(txt)
    if gm:
        grade = int(gm.group(1))
        issues = im.group(1).strip()[:120] if im else ""
        return grade, issues
    return 0, txt[:200]


class EndpointPool:
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
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def send(ep: str, img_bytes: bytes, ocr_text: str) -> tuple[float, str]:
    img_bytes = maybe_resize(img_bytes)
    img_b64 = base64.b64encode(img_bytes).decode()
    # Truncate ocr_text to keep prompt within max-model-len budget
    if len(ocr_text) > 6000:
        ocr_text = ocr_text[:6000] + "\n[…truncated…]"
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT_TMPL.format(ocr_text=ocr_text)},
            ],
        }],
        "max_tokens": 256,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(f"http://{ep}/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return elapsed, msg.get("content") or msg.get("reasoning_content") or ""


def process(pool: EndpointPool, sid: str, img_bytes: bytes, ocr_text: str) -> dict:
    last_err = None
    for attempt in range(3):
        ep = pool.get()
        try:
            elapsed, txt = send(ep, img_bytes, ocr_text)
            grade, issues = parse_response(txt)
            return {
                "id": sid,
                "ver_elapsed_s": round(elapsed, 2),
                "ver_grade": grade,
                "ver_issues": issues,
                "ver_raw": txt[:500],
                "ver_endpoint": ep,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 0:
                print(f"  [err] {sid} attempt={attempt} ep={ep}: {last_err[:200]}", flush=True)
            pool.report_failure(ep)
    return {
        "id": sid, "ver_elapsed_s": -1.0, "ver_grade": 0,
        "ver_issues": str(last_err)[:120], "ver_raw": "",
        "ver_endpoint": "",
    }


def run_shard(image_shard: Path, ocr_table_by_id: dict, gate_keep_ids: set,
               pool: EndpointPool, out_dir: Path, parallelism: int):
    out_path = out_dir / image_shard.name
    if out_path.exists():
        print(f"[skip] {image_shard.name} (already exists)")
        return
    t0 = time.time()
    table = pq.read_table(image_shard, columns=["id", "image"])
    rows_all = table.to_pylist()
    n_total = len(rows_all)
    rows = []
    for r in rows_all:
        sid = r["id"]
        if sid not in gate_keep_ids:
            continue
        ocr = ocr_table_by_id.get(sid)
        if not ocr:
            continue
        if ocr.get("ocr_no_text") or not ocr.get("ocr_text"):
            continue
        rows.append((sid, r["image"], ocr["ocr_text"]))
    n = len(rows)
    if n == 0:
        print(f"[{image_shard.name}] 0/{n_total} rows after gate+ocr filter — writing empty parquet")
        pq.write_table(pa.Table.from_pylist([], schema=pa.schema([
            pa.field("id", pa.string()), pa.field("ver_elapsed_s", pa.float64()),
            pa.field("ver_grade", pa.int32()), pa.field("ver_issues", pa.string()),
            pa.field("ver_raw", pa.string()), pa.field("ver_endpoint", pa.string()),
        ])), out_path)
        return
    print(f"[{image_shard.name}] starting on {n}/{n_total} rows (gate-KEEP w/ OCR text), parallelism={parallelism}")
    results: list[dict] = [None] * n  # type: ignore
    with ThreadPoolExecutor(max_workers=parallelism) as ex:
        fut2idx = {ex.submit(process, pool, sid, img, ocr_text): i for i, (sid, img, ocr_text) in enumerate(rows)}
        done = 0
        for fut in as_completed(fut2idx):
            idx = fut2idx[fut]
            results[idx] = fut.result()
            done += 1
            if done % 200 == 0 or done == n:
                elapsed = time.time() - t0
                rps = done / elapsed if elapsed > 0 else 0
                print(f"  [{image_shard.name}] {done}/{n} ({rps:.1f} req/s)")
    rs_table = pa.Table.from_pylist(results)
    pq.write_table(rs_table, out_path)
    elapsed = time.time() - t0
    grade_dist = Counter(r["ver_grade"] for r in results)
    print(f"[{image_shard.name}] done in {elapsed:.0f}s ({n/elapsed:.1f} req/s) — grades {dict(grade_dist)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints-dir", required=True)
    ap.add_argument("--image-dir", default="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/smithsonian/smithsonian_cleaned5")
    ap.add_argument("--ocr-dir",   default="/capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_full")
    ap.add_argument("--gate-dir",  default="/capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_gate_platinum")
    ap.add_argument("--output-dir", default="/capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_verified")
    ap.add_argument("--parallelism", type=int, default=200)
    ap.add_argument("--expected-endpoints", type=int, default=72)
    ap.add_argument("--shards", nargs="*")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load gate KEEPs across all gate shards
    gate_shards = sorted(Path(args.gate_dir).glob("*.parquet"))
    print(f"loading gate decisions from {len(gate_shards)} shards…")
    gate_keep_ids = set()
    for s in gate_shards:
        t = pq.read_table(s, columns=["id","gate_decision"])
        ids = t["id"].to_pylist()
        decs = t["gate_decision"].to_pylist()
        for sid, d in zip(ids, decs):
            if d == "KEEP":
                gate_keep_ids.add(sid)
    print(f"  → {len(gate_keep_ids):,} gate-KEEP ids")

    # Load OCR text by id (memory ok — 27K rows × ~1KB avg)
    ocr_shards = sorted(Path(args.ocr_dir).glob("*.parquet"))
    print(f"loading OCR text from {len(ocr_shards)} shards…")
    ocr_by_id = {}
    for s in ocr_shards:
        t = pq.read_table(s, columns=["id","ocr_text","ocr_no_text"]).to_pylist()
        for r in t:
            if r["id"] in gate_keep_ids:
                ocr_by_id[r["id"]] = r
    print(f"  → {len(ocr_by_id):,} OCR rows for gate-KEEP ids")

    endpoints = discover_endpoints(Path(args.endpoints_dir), expected=args.expected_endpoints, timeout=300)
    print(f"\nready with {len(endpoints)} endpoints; starting at {time.strftime('%H:%M:%S')}")
    pool = EndpointPool(endpoints)

    image_shards = sorted(Path(args.image_dir).glob("*.parquet"))
    if args.shards:
        image_shards = [s for s in image_shards if s.name in args.shards]
    print(f"will process {len(image_shards)} image shards")

    t0 = time.time()
    for shard in image_shards:
        run_shard(shard, ocr_by_id, gate_keep_ids, pool, out_dir, args.parallelism)
    wall = time.time() - t0
    print(f"\n=== ALL DONE in {wall:.0f}s ===")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
