"""BLIP3-style grounded caption pass over Mitsua/art-museums-pd-440k tars.

For each image, emits a Level-2 BLIP3-Grounding-style description:

    This image is a 17th-century Dutch still life depicting fruit and game.
    The image contains <object>silver platter</object><bbox>[0.12, 0.45][0.78, 0.82]</bbox>,
    <object>roasted fowl</object><bbox>[0.34, 0.40][0.62, 0.68]</bbox>, ...

Architecture mirrors run_ocr.py / run_quality_gate.py:
    * pool over the existing 72 Qwen3.6-27B vLLM endpoints
    * tar-stream images; one parquet output per source tar shard
    * resume-safe: existing per-shard parquet outputs are skipped

Output columns (per source tar):
    id, title, source, license, url,
    gc_elapsed_s, gc_n_objects, gc_caption (raw model text), gc_endpoint
"""
import argparse
import base64
import io
import json
import re
import tarfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from PIL import Image

MAX_IMG_DIM = 1280
MODEL = "Qwen/Qwen3.6-27B-xyixuan"

PROMPT = (
    "Look at this image and produce a grounded description in this EXACT format:\n\n"
    "<one or two sentences describing the overall image>\n"
    "The image contains <object>NAME1</object><bbox>[x1, y1][x2, y2]</bbox>, "
    "<object>NAME2</object><bbox>[x1, y1][x2, y2]</bbox>, ...\n\n"
    "Rules:\n"
    "- Coordinates are normalised 0.0-1.0 (x = horizontal, y = vertical from top)\n"
    "- Bounding boxes use [top-left][bottom-right] = [x1, y1][x2, y2]\n"
    "- List 5-12 distinct visible objects, prioritising the most prominent ones\n"
    "- Object names are short noun phrases (1-4 words), lowercase\n"
    "- The first sentence(s) give artistic/historical context (style, period, subject)\n"
    "- Do not invent objects that are not visible\n"
    "- No preamble, no markdown, no list bullets — output only the grounded description"
)

OBJ_RE = re.compile(r"<object>([^<]+)</object>\s*<bbox>\[([^\]]+)\]\[([^\]]+)\]</bbox>")
COORD_RE = re.compile(r"\s*([+-]?[0-9]*\.?[0-9]+)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\s*")


def parse_objects(txt: str) -> int:
    """Count well-formed <object>…</object><bbox>[x,y][x,y]</bbox> matches."""
    n = 0
    for m in OBJ_RE.finditer(txt):
        if COORD_RE.fullmatch(m.group(2)) and COORD_RE.fullmatch(m.group(3)):
            n += 1
    return n


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
        "max_tokens": 768,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(f"http://{ep}/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return elapsed, msg.get("content") or msg.get("reasoning_content") or ""


def process(pool: EndpointPool, sample: dict) -> dict:
    last_err = None
    for attempt in range(3):
        ep = pool.get()
        try:
            elapsed, txt = send(ep, sample["image_bytes"])
            n_obj = parse_objects(txt)
            return {
                "id":            sample["id"],
                "title":         sample.get("title", ""),
                "source":        sample.get("source", ""),
                "license":       sample.get("license", ""),
                "url":           sample.get("url", ""),
                "gc_elapsed_s":  round(elapsed, 2),
                "gc_n_objects":  n_obj,
                "gc_caption":    txt,
                "gc_endpoint":   ep,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 0:
                print(f"  [err] {sample['id'][:50]} attempt={attempt} ep={ep}: {last_err[:200]}", flush=True)
            pool.report_failure(ep)
    return {
        "id":            sample["id"],
        "title":         sample.get("title", ""),
        "source":        sample.get("source", ""),
        "license":       sample.get("license", ""),
        "url":           sample.get("url", ""),
        "gc_elapsed_s":  -1.0,
        "gc_n_objects":  0,
        "gc_caption":    f"[ERROR] {last_err}",
        "gc_endpoint":   "",
    }


def iter_tar_samples(tar_path: Path, limit: int | None = None):
    """Yield {id, image_bytes, title, source, license, url} per image in a tar."""
    pending = {}
    n = 0
    with tarfile.open(tar_path, "r") as tar:
        for ti in tar:
            if not ti.isfile(): continue
            name = ti.name
            stem, _, ext = name.rpartition(".")
            if not stem: continue
            f = tar.extractfile(ti)
            if f is None: continue
            data = f.read()
            entry = pending.setdefault(stem, {})
            if ext == "jpg":
                entry["image_bytes"] = data
            elif ext == "json":
                try:
                    j = json.loads(data)
                    entry["title"]   = j.get("Title", "")
                    entry["source"]  = j.get("source", "")
                    entry["license"] = j.get("License", "")
                    entry["url"]     = j.get("URL", "")
                except Exception:
                    pass
            elif ext == "txt":
                pass  # skip multilingual recap captions
            if "image_bytes" in entry and "title" in entry:
                yield {"id": stem, **entry}
                pending.pop(stem, None)
                n += 1
                if limit and n >= limit:
                    return
    # flush leftover entries that have an image but no JSON
    for stem, entry in pending.items():
        if "image_bytes" in entry:
            yield {"id": stem, "title": "", "source": "", "license": "", "url": "", **entry}


def run_shard(tar_path: Path, pool: EndpointPool, out_dir: Path, parallelism: int, limit: int | None):
    out_path = out_dir / f"{tar_path.stem}.parquet"
    if out_path.exists():
        print(f"[skip] {tar_path.name} (already exists)")
        return
    t0 = time.time()
    samples = list(iter_tar_samples(tar_path, limit=limit))
    n = len(samples)
    print(f"[{tar_path.name}] starting on {n} images, parallelism={parallelism}")
    results: list[dict] = [None] * n
    with ThreadPoolExecutor(max_workers=parallelism) as ex:
        fut2idx = {ex.submit(process, pool, s): i for i, s in enumerate(samples)}
        done = 0
        for fut in as_completed(fut2idx):
            idx = fut2idx[fut]
            results[idx] = fut.result()
            done += 1
            if done % 200 == 0 or done == n:
                elapsed = time.time() - t0
                rps = done / elapsed if elapsed > 0 else 0
                avg_obj = sum(r["gc_n_objects"] for r in results if r) / max(done, 1)
                print(f"  [{tar_path.name}] {done}/{n} ({rps:.1f} req/s, {avg_obj:.1f} obj/img)")
    pq.write_table(pa.Table.from_pylist(results), out_path, compression="zstd")
    elapsed = time.time() - t0
    nobj_dist = Counter(r["gc_n_objects"] for r in results)
    print(f"[{tar_path.name}] done in {elapsed:.0f}s ({n/elapsed:.1f} req/s) — n_objects {dict(sorted(nobj_dist.items()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints-dir", required=True)
    ap.add_argument("--input-dir", default="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/hf___Mitsua___art-museums-pd-440k___recap")
    ap.add_argument("--output-dir", default="/capstor/scratch/cscs/xyixuan/recon/qwen_artmuseums_grounded")
    ap.add_argument("--parallelism", type=int, default=200)
    ap.add_argument("--expected-endpoints", type=int, default=72)
    ap.add_argument("--shards", nargs="*", help="subset of shard names; default: all")
    ap.add_argument("--limit-per-shard", type=int, help="cap rows per shard (smoke testing)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoints = discover_endpoints(Path(args.endpoints_dir), expected=args.expected_endpoints, timeout=300)
    print(f"\nready with {len(endpoints)} endpoints; starting at {time.strftime('%H:%M:%S')}")
    pool = EndpointPool(endpoints)

    tars = sorted(Path(args.input_dir).glob("*.tar"))
    if args.shards:
        tars = [t for t in tars if t.name in args.shards]
    print(f"will process {len(tars)} shards")

    t0 = time.time()
    for tar in tars:
        run_shard(tar, pool, out_dir, args.parallelism, args.limit_per_shard)
    wall = time.time() - t0
    print(f"\n=== ALL DONE in {wall:.0f}s ===")
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
