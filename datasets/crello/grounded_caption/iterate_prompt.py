"""Prompt iteration harness for Crello BLIP3-Grounding-style captioning.

Architecture:
    * Read N designs from one Crello shard
    * Pre-compute ground-truth element list (bbox, type, text) per design
    * Run each of K prompt variants against the same designs in parallel
    * Print quality metrics + 1 sample per variant for inspection

Quality metrics per variant:
    * fluency        : avg caption length (chars)
    * structure_ok   : fraction of outputs that match BLIP3 schema
                      (opening sentence + "The design contains" + ≥3 grounded objects)
    * gt_coord_match : fraction of <bbox>...</bbox> tokens that match ground-truth
                      coordinates exactly (rounded to 3 decimal places)
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


MAX_COVERAGE = 0.95
MIN_COVERAGE = 0.0005


def gt_elements(row):
    """Return list of (label_hint, x1, y1, x2, y2) per element."""
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
            x1,y1,x2,y2 = normalise_bbox(lefts[i], tops[i], widths[i], heights[i], cw, ch)
        except (IndexError, TypeError):
            continue
        if x2 <= x1+1e-3 or y2 <= y1+1e-3:
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


# ─────────────────────── Prompt variants ──────────────────────────────────────

def prompt_A_freeform(_elems):
    return (
        "This is a graphic-design template. Produce a BLIP3-Grounding-style "
        "description in this EXACT format:\n\n"
        "<one or two sentences describing the overall design>\n"
        "The design contains <object>NAME1</object><bbox>[x1, y1][x2, y2]</bbox>, "
        "<object>NAME2</object><bbox>[x1, y1][x2, y2]</bbox>, ...\n\n"
        "Rules:\n"
        "- Coordinates are normalised 0.0-1.0\n"
        "- 5-12 distinct objects, lowercase short noun phrases\n"
        "- Output only the grounded description, no preamble"
    )


def prompt_B_provided_bboxes(elems):
    """Provide GT bboxes; Qwen only chooses semantic labels."""
    lines = []
    for i, (hint, x1, y1, x2, y2) in enumerate(elems):
        lines.append(f"  [{i}] {hint} at <bbox>[{x1:.3f}, {y1:.3f}][{x2:.3f}, {y2:.3f}]</bbox>")
    listing = "\n".join(lines)
    return (
        "This is a graphic-design template. Below is the ground-truth element "
        "list with bounding boxes (normalised 0.0-1.0):\n\n"
        f"{listing}\n\n"
        "Look at the image and write a BLIP3-Grounding-style description. "
        "Format:\n\n"
        "<one or two opening sentences describing the overall design — "
        "what it advertises or communicates, the visual style, mood>\n"
        "The design contains <object>SEMANTIC_LABEL</object><bbox>[x1, y1][x2, y2]</bbox>, "
        "<object>SEMANTIC_LABEL</object><bbox>[x1, y1][x2, y2]</bbox>, ...\n\n"
        "Rules:\n"
        "- Use the bounding boxes EXACTLY as given above — do not modify numbers\n"
        "- Replace each element's hint with a richer semantic label:\n"
        "    * text elements → keep the text content but add role (e.g. "
        "'headline \"SUMMER SALE\"', 'body text \"...\"', 'call-to-action button \"Shop Now\"')\n"
        "    * image elements → describe what is depicted in the embedded image\n"
        "    * graphic elements → describe their decorative role\n"
        "- Include ALL ground-truth elements, in the same order\n"
        "- Output only the grounded description, no preamble"
    )


def prompt_D_rich(elems):
    """Variant B but ask for substantially richer prose AND richer labels."""
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


def prompt_C_template_fill(elems):
    """Strict template fill — only fills in semantic labels."""
    template_lines = []
    for i, (hint, x1, y1, x2, y2) in enumerate(elems):
        template_lines.append(
            f"<object>{{LABEL_{i}}}</object><bbox>[{x1:.3f}, {y1:.3f}][{x2:.3f}, {y2:.3f}]</bbox>"
        )
    template = ", ".join(template_lines)
    hints = "\n".join(
        f"  LABEL_{i} (current hint: {hint}) — bbox [{x1:.3f}, {y1:.3f}][{x2:.3f}, {y2:.3f}]"
        for i, (hint, x1, y1, x2, y2) in enumerate(elems)
    )
    return (
        "Look at the image. Below is the layout of all elements with their "
        "ground-truth bounding boxes. Your task: fill in a richer semantic "
        "label for each LABEL_N, then emit the completed BLIP3-Grounding "
        "description.\n\n"
        f"Element hints:\n{hints}\n\n"
        "Output format:\n\n"
        "<one or two opening sentences describing the overall design>\n"
        f"The design contains {template}.\n\n"
        "Rules:\n"
        "- DO NOT change any bbox numbers — they are ground truth\n"
        "- For text elements, include the actual text content in the label "
        "(use quotes: 'headline \"SAVE 20%\"')\n"
        "- For image elements, describe what is shown in the embedded image\n"
        "- For graphic elements, describe their decorative role\n"
        "- Output only the completed description, no preamble"
    )


VARIANTS = {
    "A_freeform":         prompt_A_freeform,
    "B_provided_bboxes":  prompt_B_provided_bboxes,
    "C_template_fill":    prompt_C_template_fill,
    "D_rich":             prompt_D_rich,
}


# ─────────────────────── Endpoint pool ────────────────────────────────────────


def discover_endpoints(endpoints_dir, expected=72):
    eps = sorted(set(p.read_text().strip() for p in Path(endpoints_dir).glob("*.endpoint")))
    healthy = []
    for ep in eps:
        try:
            if requests.get(f"http://{ep}/v1/models", timeout=3).status_code == 200:
                healthy.append(ep)
        except Exception:
            pass
    print(f"  {len(healthy)}/{len(eps)} endpoints healthy")
    return healthy


def maybe_resize(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    if max(img.size) <= MAX_IMG_DIM:
        return img_bytes
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
        "max_tokens": 1024,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"http://{ep}/v1/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def evaluate(out_text, gt_elems):
    """Return (n_objects, structure_ok, gt_coord_match_frac)."""
    matches = list(OBJ_RE.finditer(out_text))
    n = len(matches)
    has_open = "the design contains" in out_text.lower() or "the image contains" in out_text.lower()
    structure_ok = has_open and n >= 3
    if not matches:
        return n, structure_ok, 0.0
    gt_set = set(
        (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3))
        for (_, x1, y1, x2, y2) in gt_elems
    )
    matched = 0
    for m in matches:
        try:
            a = COORD_RE.fullmatch(m.group(2))
            b = COORD_RE.fullmatch(m.group(3))
            if not (a and b): continue
            x1, y1 = float(a.group(1)), float(a.group(2))
            x2, y2 = float(b.group(1)), float(b.group(2))
            if (round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)) in gt_set:
                matched += 1
        except Exception:
            pass
    return n, structure_ok, matched / n if n > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints-dir", required=True)
    ap.add_argument("--shard", default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/hf___cyberagent___crello/data/train-00000-of-00031.parquet")
    ap.add_argument("--n-designs", type=int, default=10)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    ap.add_argument("--show-samples", type=int, default=2)
    args = ap.parse_args()

    endpoints = discover_endpoints(args.endpoints_dir)
    print(f"loading {args.n_designs} designs from {Path(args.shard).name}")
    table = pq.read_table(args.shard, columns=[
        "id","preview","canvas_width","canvas_height","title",
        "format","category","length","type","left","top","width","height","text"
    ])
    rows = table.slice(0, args.n_designs).to_pylist()
    designs = []
    for r in rows:
        prev = r.get("preview") or {}
        b = prev.get("bytes") if isinstance(prev, dict) else None
        if not b: continue
        elems = gt_elements(r)
        if len(elems) < 3: continue
        designs.append({"id": r["id"], "image_bytes": b, "elements": elems,
                        "title": r.get("title") or ""})
    print(f"  → {len(designs)} valid designs (≥3 GT elements)")

    results_per_variant = {v: [] for v in args.variants}

    def task(idx, variant_name, design):
        prompt = VARIANTS[variant_name](design["elements"])
        ep = endpoints[idx % len(endpoints)]
        try:
            t0 = time.time()
            txt = send(ep, design["image_bytes"], prompt)
            elapsed = time.time() - t0
        except Exception as e:
            return variant_name, design["id"], "", -1.0, str(e)[:200]
        return variant_name, design["id"], txt, elapsed, None

    print(f"\nrunning {len(args.variants)} variants × {len(designs)} designs = "
          f"{len(args.variants)*len(designs)} requests…")
    with ThreadPoolExecutor(max_workers=72) as ex:
        futs = []
        idx = 0
        for v in args.variants:
            for d in designs:
                futs.append(ex.submit(task, idx, v, d))
                idx += 1
        for fut in as_completed(futs):
            v, did, txt, elapsed, err = fut.result()
            d = next(d for d in designs if d["id"] == did)
            n, ok, frac = evaluate(txt, d["elements"]) if txt else (0, False, 0.0)
            results_per_variant[v].append({
                "id": did, "txt": txt, "elapsed": elapsed,
                "n_objects": n, "structure_ok": ok, "gt_match": frac,
                "n_gt": len(d["elements"]),
                "title": d["title"], "err": err,
            })

    # ─── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"{'variant':<25s} {'avg_n':>7s} {'struct%':>8s} {'gt_match%':>10s} {'avg_chars':>10s} {'avg_s':>7s}")
    print("="*72)
    for v in args.variants:
        rs = results_per_variant[v]
        n = len(rs)
        avg_n = sum(r["n_objects"] for r in rs) / n if n else 0
        struct = sum(1 for r in rs if r["structure_ok"]) / n if n else 0
        gt = sum(r["gt_match"] for r in rs) / n if n else 0
        chars = sum(len(r["txt"]) for r in rs) / n if n else 0
        sec = sum(r["elapsed"] for r in rs if r["elapsed"]>0) / max(1, sum(1 for r in rs if r["elapsed"]>0))
        print(f"{v:<25s} {avg_n:>7.1f} {struct*100:>7.0f}% {gt*100:>9.0f}% {chars:>10.0f} {sec:>7.2f}")

    # Show samples
    for v in args.variants:
        print(f"\n{'─'*72}\nVARIANT: {v}\n{'─'*72}")
        for r in results_per_variant[v][:args.show_samples]:
            print(f"\n[id={r['id'][:24]}] n_obj={r['n_objects']}/{r['n_gt']}  gt_match={r['gt_match']:.0%}  struct_ok={r['structure_ok']}")
            print(f"title: {r['title']}")
            print(r['txt'][:1200])
            if len(r['txt']) > 1200: print("…[truncated]")


if __name__ == "__main__":
    main()
