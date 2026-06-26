"""BLIP3-Grounding Level-2 captioning for cyberagent/crello.

Strategy (after prompt iteration — Variant B):
    * Coordinates come from Crello's GROUND-TRUTH designer-authored bboxes
      (left/top/width/height per element, normalised to [0.0, 1.0])
    * Qwen3.6-27B receives the image + the pre-computed ground-truth element
      list, and only contributes SEMANTIC LABELS (e.g. 'headline "SAVE 20%"',
      'product photo of a leather handbag', 'decorative wave shape')
    * Qwen is instructed to use the bboxes EXACTLY as given — verified at
      100% gt-match rate on the 10-design smoke test

Output schema (per source parquet shard, joined to crello by id):
    id, title, canvas_width, canvas_height, n_elements,
    image (preview JPEG bytes),
    grounded_text (BLIP3-Grounding Level-2 description, GT coords),
    gc_elapsed_s, gc_n_objects, gc_endpoint
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

MAX_IMG_DIM = 1280
MODEL = "Qwen/Qwen3.6-27B-xyixuan"

OBJ_RE = re.compile(r"<object>([^<]+)</object>\s*<bbox>\[([^\]]+)\]\[([^\]]+)\]</bbox>")
COORD_RE = re.compile(r"\s*([+-]?[0-9]*\.?[0-9]+)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\s*")


def normalise_bbox(left, top, w, h, cw, ch):
    if cw <= 0 or ch <= 0: return 0.0, 0.0, 0.0, 0.0
    return (max(0.0, left/cw), max(0.0, top/ch),
            min(1.0, (left+w)/cw), min(1.0, (top+h)/ch))


# Filter elements that have weak grounding signal — bbox covering nearly
# the entire canvas teaches the model nothing about *where* something is.
MAX_COVERAGE = 0.95   # drop elements whose bbox covers ≥95% of canvas area
MIN_COVERAGE = 0.0005 # drop elements smaller than 0.05% (mostly noise / sub-pixel)


def gt_elements(row):
    """Return ordered list of (label_hint, x1, y1, x2, y2) per element,
    filtering full-canvas backgrounds and noise-sized elements."""
    cw = int(row.get("canvas_width") or 0)
    ch = int(row.get("canvas_height") or 0)
    types  = row.get("type") or []
    lefts  = row.get("left") or []
    tops   = row.get("top") or []
    widths = row.get("width") or []
    heights= row.get("height") or []
    texts  = row.get("text") or []
    out = []
    for i in range(int(row.get("length") or 0)):
        try:
            x1, y1, x2, y2 = normalise_bbox(lefts[i], tops[i], widths[i], heights[i], cw, ch)
        except (IndexError, TypeError):
            continue
        if x2 <= x1 + 1e-3 or y2 <= y1 + 1e-3:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area >= MAX_COVERAGE or area < MIN_COVERAGE:
            continue
        t = (texts[i] if i < len(texts) else "") or ""
        t = t.strip().replace("\n", " ")
        if t:
            hint = f'text "{t[:60]}"' if len(t) <= 60 else f'text "{t[:60]}…"'
        elif int(types[i]) in (0, 3):
            hint = "image element"
        else:
            hint = "graphic element"
        out.append((hint, x1, y1, x2, y2))
    return out


def build_prompt(elems):
    """D_rich prompt — picked from prompt iteration:
        struct=100%, gt_match=100%, avg 1631 chars, ~6.9 grounded objects.

    Asks Qwen for a 4-6 sentence opening (format / palette / layout /
    typography / embedded subjects / audience) followed by the BLIP3
    grounded-element line using the GROUND-TRUTH bboxes verbatim and
    Qwen-authored richer semantic labels.
    """
    lines = []
    for i, (hint, x1, y1, x2, y2) in enumerate(elems):
        lines.append(f"  [{i}] {hint} at <bbox>[{x1:.3f}, {y1:.3f}][{x2:.3f}, {y2:.3f}]</bbox>")
    listing = "\n".join(lines)
    return (
        "You are writing a richly-detailed BLIP3-Grounding-style description "
        "of a graphic-design template for a vision-language training corpus. "
        "Below is the ground-truth element list with bounding boxes "
        "(normalised 0.0-1.0):\n\n"
        f"{listing}\n\n"
        "Output format (TWO parts):\n\n"
        "PART 1 — Opening prose (4-6 sentences, dense):\n"
        "  • Sentence 1: what the design advertises/communicates and its format "
        "(e.g. Instagram square ad, product flyer, magazine cover, presentation "
        "slide).\n"
        "  • Sentence 2-3: visual style and mood (palette in concrete colour "
        "names, illustration vs photograph, modern vs vintage, energetic vs "
        "calm), and the layout strategy (centred, split, asymmetric, grid).\n"
        "  • Sentence 4-5: typography and treatment (which fonts feel serif/"
        "sans-serif/script, which words are emphasised, hierarchy from headline "
        "to body to CTA), plus what's depicted in any embedded photographs or "
        "illustrations (subjects, setting, action).\n"
        "  • Sentence 6: who the design is FOR (target audience) or what action "
        "it asks the viewer to take.\n\n"
        "PART 2 — One line listing every grounded element. EVERY element MUST "
        "be wrapped in `<object>...</object>` followed immediately by `<bbox>"
        "[x1, y1][x2, y2]</bbox>`. No bare text, no missing tags. Format:\n\n"
        "  The design contains <object>LABEL_1</object><bbox>[x1, y1][x2, y2]</bbox>, "
        "<object>LABEL_2</object><bbox>[x1, y1][x2, y2]</bbox>, ...\n\n"
        "CONCRETE EXAMPLE of a single element entry (note both opening AND closing tags):\n"
        '  <object>navy serif headline "WATERFRONT"</object><bbox>[0.060, 0.096][0.448, 0.142]</bbox>\n\n'
        "Semantic labels MUST be richer than the hints. Examples of strong "
        "labels:\n"
        "  • 'bold red sans-serif headline \"SUMMER SALE\"' (not 'text \"SUMMER SALE\"')\n"
        "  • 'photograph of a steaming coffee cup on a wooden table' (not 'image element')\n"
        "  • 'curved beige decorative wave in the lower-right corner' (not 'graphic element')\n"
        "  • 'pink rounded call-to-action button reading \"SHOP NOW\"'\n"
        "  • 'small navy heart icon'\n\n"
        "Strict rules:\n"
        "- Use the bounding boxes EXACTLY as given above — do not modify numbers\n"
        "- Include ALL ground-truth elements in the order given\n"
        "- EVERY element entry needs the full `<object>LABEL</object><bbox>...</bbox>` structure\n"
        "- Each label is a noun phrase (3-12 words), lowercase except for proper nouns/quoted text\n"
        "- For text elements, keep the text content quoted in the label\n"
        "- For image elements, describe content + style (e.g. 'illustration of', 'photograph of', 'flat-icon of')\n"
        "- Output ONLY: PART 1 prose then a blank line then PART 2 single line; no preamble, no markdown headers"
    )


def count_gt_match(text: str, gt_elems: list) -> tuple[int, int]:
    """Return (n_objects_in_output, n_with_gt_coord_match)."""
    matches = list(OBJ_RE.finditer(text))
    if not matches: return 0, 0
    gt_set = set(
        (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
        for (_, x1, y1, x2, y2) in gt_elems
    )
    matched = 0
    for m in matches:
        a = COORD_RE.fullmatch(m.group(2))
        b = COORD_RE.fullmatch(m.group(3))
        if not (a and b): continue
        try:
            x1, y1 = float(a.group(1)), float(a.group(2))
            x2, y2 = float(b.group(1)), float(b.group(2))
            if (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)) in gt_set:
                matched += 1
        except Exception:
            pass
    return len(matches), matched


# ─────────────────────── Endpoint pool ────────────────────────────────────────


class EndpointPool:
    def __init__(self, endpoints):
        self.endpoints = list(endpoints); self.failures = Counter()
        self.lock = Lock(); self.idx = 0

    def get(self):
        with self.lock:
            if not self.endpoints: raise RuntimeError("no healthy endpoints left")
            ep = self.endpoints[self.idx % len(self.endpoints)]; self.idx += 1
            return ep

    def report_failure(self, ep):
        with self.lock:
            self.failures[ep] += 1
            if self.failures[ep] >= 3 and ep in self.endpoints:
                print(f"  [pool] dropping {ep}"); self.endpoints.remove(ep)


def discover_endpoints(endpoints_dir, expected=72, timeout=300):
    deadline = time.time() + timeout
    eps = []
    while time.time() < deadline:
        eps = sorted(set(p.read_text().strip() for p in Path(endpoints_dir).glob("*.endpoint")))
        if len(eps) >= expected: break
        time.sleep(15)
    healthy = []
    for ep in eps:
        try:
            if requests.get(f"http://{ep}/v1/models", timeout=5).status_code == 200:
                healthy.append(ep)
        except Exception: pass
    print(f"  {len(healthy)}/{len(eps)} endpoints healthy")
    return healthy


def maybe_resize(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    if max(img.size) <= MAX_IMG_DIM: return img_bytes
    img.thumbnail((MAX_IMG_DIM, MAX_IMG_DIM), Image.Resampling.LANCZOS)
    if img.mode != "RGB": img = img.convert("RGB")
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90); return buf.getvalue()


def send(ep, img_bytes, prompt):
    img_bytes = maybe_resize(img_bytes)
    img_b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role":"user","content":[
            {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}},
            {"type":"text","text":prompt},
        ]}],
        "max_tokens": 1500,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(f"http://{ep}/v1/chat/completions", json=payload, timeout=300)
    elapsed = time.time() - t0
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return elapsed, msg.get("content") or msg.get("reasoning_content") or ""


def process(pool, sample):
    last_err = None
    prompt = build_prompt(sample["elements"])
    for attempt in range(3):
        ep = pool.get()
        try:
            elapsed, txt = send(ep, sample["image_bytes"], prompt)
            n_obj, n_match = count_gt_match(txt, sample["elements"])
            return {
                "id":             sample["id"],
                "title":          sample["title"],
                "canvas_width":   sample["canvas_width"],
                "canvas_height":  sample["canvas_height"],
                "n_elements":     sample["n_elements"],
                "image":          sample["image_bytes"],
                "grounded_text":  txt,
                "gc_elapsed_s":   round(elapsed, 2),
                "gc_n_objects":   n_obj,
                "gc_n_gt_match":  n_match,
                "gc_endpoint":    ep,
            }
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt == 0:
                print(f"  [err] {sample['id'][:30]} ep={ep}: {last_err[:200]}", flush=True)
            pool.report_failure(ep)
    return {
        "id":             sample["id"],
        "title":          sample["title"],
        "canvas_width":   sample["canvas_width"],
        "canvas_height":  sample["canvas_height"],
        "n_elements":     sample["n_elements"],
        "image":          sample["image_bytes"],
        "grounded_text":  f"[ERROR] {last_err}",
        "gc_elapsed_s":   -1.0,
        "gc_n_objects":   0,
        "gc_n_gt_match":  0,
        "gc_endpoint":    "",
    }


# ─────────────────────── Driver ───────────────────────────────────────────────


def run_shard(shard, pool, out_dir, parallelism, limit):
    out_path = out_dir / shard.name
    if out_path.exists():
        print(f"[skip] {shard.name}"); return
    t0 = time.time()
    cols = ["id","preview","canvas_width","canvas_height","title","format",
            "category","length","type","left","top","width","height","text"]
    rows = pq.read_table(shard, columns=cols).to_pylist()
    if limit: rows = rows[:limit]

    samples = []
    skipped = 0
    for r in rows:
        prev = r.get("preview") or {}
        b = prev.get("bytes") if isinstance(prev, dict) else None
        if not b: skipped += 1; continue
        elems = gt_elements(r)
        if len(elems) < 2:
            skipped += 1; continue
        samples.append({
            "id":            r["id"],
            "title":         r.get("title") or "",
            "canvas_width":  int(r.get("canvas_width") or 0),
            "canvas_height": int(r.get("canvas_height") or 0),
            "n_elements":    int(r.get("length") or 0),
            "elements":      elems,
            "image_bytes":   b,
        })
    n = len(samples)
    print(f"[{shard.name}] starting on {n} designs ({skipped} skipped), parallelism={parallelism}")

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
                avg_match = sum(r["gc_n_gt_match"] / max(r["gc_n_objects"], 1) for r in results if r) / done
                print(f"  [{shard.name}] {done}/{n} ({rps:.1f} req/s, gt_match={avg_match:.0%})")

    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("canvas_width", pa.int64()),
        pa.field("canvas_height", pa.int64()),
        pa.field("n_elements", pa.int64()),
        pa.field("image", pa.binary()),
        pa.field("grounded_text", pa.string()),
        pa.field("gc_elapsed_s", pa.float64()),
        pa.field("gc_n_objects", pa.int64()),
        pa.field("gc_n_gt_match", pa.int64()),
        pa.field("gc_endpoint", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(results, schema=schema), out_path,
                   compression="zstd")
    elapsed = time.time() - t0
    nobj_dist = Counter(r["gc_n_objects"] for r in results)
    print(f"[{shard.name}] done in {elapsed:.0f}s ({n/elapsed:.1f} req/s) — n_objects {dict(sorted(nobj_dist.items())[:8])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints-dir", required=True)
    ap.add_argument("--input-dir", default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/hf___cyberagent___crello/data")
    ap.add_argument("--output-dir", default="/capstor/scratch/cscs/xyixuan/recon/qwen_crello_grounded")
    ap.add_argument("--parallelism", type=int, default=200)
    ap.add_argument("--expected-endpoints", type=int, default=72)
    ap.add_argument("--shards", nargs="*")
    ap.add_argument("--limit-per-shard", type=int)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    endpoints = discover_endpoints(Path(args.endpoints_dir), expected=args.expected_endpoints)
    print(f"\nready with {len(endpoints)} endpoints; starting at {time.strftime('%H:%M:%S')}")
    pool = EndpointPool(endpoints)

    shards = sorted(Path(args.input_dir).glob("*.parquet"))
    if args.shards:
        shards = [s for s in shards if s.name in args.shards]
    print(f"will process {len(shards)} shards")

    t0 = time.time()
    for s in shards:
        run_shard(s, pool, out_dir, args.parallelism, args.limit_per_shard)
    print(f"\n=== ALL DONE in {time.time()-t0:.0f}s ===  outputs in {out_dir}")


if __name__ == "__main__":
    main()
