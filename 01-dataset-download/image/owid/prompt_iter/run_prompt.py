"""Test a single prompt against staged OWID samples and dump outputs.

Usage:
    python run_prompt.py --prompt-file prompt_v1.txt --tag v1
"""
import argparse
import asyncio
import base64
import json
import time
from pathlib import Path

import httpx


ENDPOINT = "http://172.28.32.28:30000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-397B-A17B-xyixuan"


async def caption(client, system, user, image_bytes, temperature=0.3):
    img_b64 = base64.b64encode(image_bytes).decode()
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": user},
            ]},
        ],
        "max_tokens": 1200,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = await client.post(ENDPOINT, json=body, timeout=600.0)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], time.time() - t0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--user", default="Caption this chart following the rules in the system prompt.")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--metadata-aware", action="store_true")
    args = ap.parse_args()

    system = Path(args.prompt_file).read_text().strip()
    samples = sorted(Path(".").glob("sample_*.png"))
    out_dir = Path(f"out_{args.tag}")
    out_dir.mkdir(exist_ok=True)

    # Load note from grapher parquet for metadata-aware mode
    import polars as pl
    df = pl.read_parquet(
        "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts/grapher_charts_standalone.parquet",
        columns=["slug", "note"])
    slug_to_note = dict(zip(df["slug"].to_list(), df["note"].to_list()))

    _MD_LINK = __import__("re").compile(r"\[([^\]]+)\]\(#?[^\)]+\)")
    def _clean(s):
        s = _MD_LINK.sub(lambda m: m.group(1), s or "")
        return s.strip()

    def build_user_prompt(meta):
        title = _clean(meta.get("title"))
        subtitle = _clean(meta.get("subtitle"))
        note = _clean(slug_to_note.get(meta["slug"]))
        ctx = ["Chart metadata (use verbatim for the title and subtitle lines; integrate the note's factual content into the paragraph):"]
        ctx.append(f"title: {title}")
        if subtitle:
            ctx.append(f"subtitle: {subtitle}")
        if note:
            ctx.append(f"note: {note}")
        ctx.append("")
        ctx.append(args.user)
        return "\n".join(ctx)

    sem = asyncio.Semaphore(16)
    async def one(client, img):
        async with sem:
            meta = json.loads(img.with_suffix(".meta.json").read_text())
            user_p = build_user_prompt(meta) if args.metadata_aware else args.user
            try:
                txt, dt = await caption(client, system, user_p, img.read_bytes(), args.temperature)
                rec = {"slug": meta["slug"], "title": meta["title"],
                       "subtitle": meta.get("subtitle",""),
                       "caption": txt, "word_count": len(txt.split()),
                       "latency_s": dt, "prompt_tag": args.tag}
                (out_dir / f"{img.stem}.json").write_text(
                    json.dumps(rec, indent=2, ensure_ascii=False))
                print(f"ok {img.stem} ({dt:.1f}s, {len(txt.split())}w)", flush=True)
            except Exception as e:
                print(f"FAIL {img.stem}: {type(e).__name__}: {str(e)[:150]}", flush=True)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=32)) as client:
        await asyncio.gather(*(one(client, img) for img in samples))


if __name__ == "__main__":
    asyncio.run(main())
