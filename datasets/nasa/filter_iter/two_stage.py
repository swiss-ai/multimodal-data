"""Two-stage pipeline: filter then regen.

Stage 1: filter prompt (prompts/v3.txt) -> KEEP / DROP
Stage 2a: if KEEP, regen with balanced.txt using full seed (fuse).
Stage 2b: if DROP, regen with balanced.txt using EMPTY seed (pure image caption).
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from labeled_samples import LABELS  # noqa: E402
from clean_caption import clean_caption  # noqa: E402

MAX_LONG_SIDE = 1024
MODEL = "Qwen/Qwen3.6-27B-BZji"
FILTER_PROMPT = (HERE / "prompts" / "v3.txt").read_text()
REGEN_PROMPT = (HERE / "regen_prompts" / "balanced.txt").read_text()
EMPTY_SEED_INSTR = "[No seed provided — caption the image from pixels alone. Do not invent identity, names, missions, locations, or dates.]"

VERDICT_RE = re.compile(r"VERDICT:\s*(KEEP|DROP)", re.IGNORECASE)


def find_image(nasa_id):
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        p = HERE / "images" / f"{nasa_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(nasa_id)


def encode_image(path):
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_LONG_SIDE:
        s = MAX_LONG_SIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call(endpoint, prompt_text, img_b64, thinking=False, max_tokens=None):
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
    r = requests.post(f"{endpoint}/v1/chat/completions", json=msg, timeout=300,
                      proxies={"http": None, "https": None})
    r.raise_for_status()
    return (r.json()["choices"][0]["message"].get("content") or "").strip()


def run_one(endpoint, nid, expected, seed, filter_thinking, regen_thinking):
    try:
        img_b64 = encode_image(find_image(nid))

        # Stage 1: filter
        filter_text = FILTER_PROMPT.format(caption=seed)
        f_out = call(endpoint, filter_text, img_b64, thinking=filter_thinking, max_tokens=(8192 if filter_thinking else 128))
        v = VERDICT_RE.search(f_out)
        verdict = v.group(1).upper() if v else "UNKNOWN"

        # Stage 2: regen (KEEP → full seed; DROP → empty seed instruction)
        if verdict == "KEEP":
            regen_seed = seed
        else:
            regen_seed = EMPTY_SEED_INSTR
        regen_text = REGEN_PROMPT.format(caption=regen_seed)
        caption = call(endpoint, regen_text, img_b64, thinking=regen_thinking)
        # Fallback: if thinking mode blew the budget and returned no content, retry without thinking.
        if regen_thinking and not caption.strip():
            caption = call(endpoint, regen_text, img_b64, thinking=False)

        return {
            "id": nid, "expected": expected, "seed": seed,
            "verdict": verdict, "caption": caption,
            "filter_raw": f_out, "error": None,
        }
    except Exception as e:
        return {
            "id": nid, "expected": expected, "seed": seed,
            "verdict": None, "caption": None,
            "filter_raw": "", "error": str(e),
        }


# Post-flip labels (from recon)
FLIPS = {
    "STS059-S-001": "KEEP", "ARC-1994-AC94-0109-3": "KEEP",
    "LRC-1962-B701_P-05849": "KEEP",
    "iss043e198394": "DROP", "AFRC2018-0287-278": "DROP",
    "GSFC_20171208_Archive_e001446": "KEEP", "EC00-0226-21": "DROP",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://172.28.28.36:8080")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--filter-thinking", action="store_true", help="enable thinking for filter stage (default off — cheap)")
    ap.add_argument("--regen-thinking", action="store_true", help="enable thinking for regen stage")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    samples = []
    for nid, label, cap in LABELS:
        exp = FLIPS.get(nid, label)
        samples.append((nid, exp, clean_caption(cap)))
    if args.limit:
        samples = samples[: args.limit]

    print(f"two-stage: filter_thinking={args.filter_thinking} regen_thinking={args.regen_thinking}  samples={len(samples)}  conc={args.concurrency}")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, args.endpoint, nid, exp, seed, args.filter_thinking, args.regen_thinking)
                for nid, exp, seed in samples]
        done = 0
        for f in futs:
            results.append(f.result())
            done += 1
            if done % 10 == 0 or done == len(futs):
                print(f"  [{done}/{len(futs)}] {time.time()-t0:.1f}s")

    # Filter accuracy
    tp = fp = tn = fn = unk = 0
    for r in results:
        e, p = r["expected"], r["verdict"]
        if p in (None, "UNKNOWN"):
            unk += 1; continue
        if e == "KEEP" and p == "KEEP": tp += 1
        elif e == "DROP" and p == "DROP": tn += 1
        elif e == "DROP" and p == "KEEP": fp += 1
        elif e == "KEEP" and p == "DROP": fn += 1
    total = len(results)
    print(f"\n=== FILTER ACCURACY ===")
    print(f"  correct: {tp+tn}/{total}  ({(tp+tn)/total*100:.1f}%)")
    print(f"  KEEP prec {tp}/{tp+fp}  recall {tp}/{tp+fn}")
    print(f"  DROP prec {tn}/{tn+fn}  recall {tn}/{tn+fp}")
    print(f"  parse_fails: {unk}  errors: {sum(1 for r in results if r['error'])}")

    # Caption production
    captions_n = sum(1 for r in results if (r.get("caption") or "").strip())
    print(f"\n=== CAPTION DELIVERY ===")
    print(f"  produced: {captions_n}/{total}")

    # Save
    suffix = f"_{args.tag}" if args.tag else ""
    out = HERE / f"results_two_stage{suffix}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nlog: {out}")


if __name__ == "__main__":
    main()
