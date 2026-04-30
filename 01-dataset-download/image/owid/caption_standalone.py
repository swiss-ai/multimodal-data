"""Caption all 3,487 OWID standalone grapher charts with V12 prompt.

Reads grapher_charts_standalone.parquet, sends each image+metadata to the
live Qwen 397B router, writes captions back to a new parquet.
"""
import argparse
import asyncio
import base64
import json
import re
import time
from pathlib import Path

import httpx
import polars as pl


ENDPOINT = "http://172.28.32.28:30000/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-397B-A17B-xyixuan"
_MD = re.compile(r"\[([^\]]+)\]\(#?[^\)]+\)")


def clean(s): return _MD.sub(lambda m: m.group(1), s or "").strip()


async def caption_one(client, sem, system, row, metrics, temperature=0.1):
    async with sem:
        title = clean(row.get("title"))
        subtitle = clean(row.get("subtitle"))
        note = clean(row.get("note"))
        ctx = ["Chart metadata (use verbatim for the title and subtitle lines; integrate the note's factual content into the paragraph):",
               f"title: {title}"]
        if subtitle: ctx.append(f"subtitle: {subtitle}")
        if note: ctx.append(f"note: {note}")
        ctx.append("")
        ctx.append("Caption this chart following the rules in the system prompt.")
        user = "\n".join(ctx)
        img_b64 = base64.b64encode(row["image_bytes"]).decode()
        body = {"model": MODEL, "max_tokens": 1200, "temperature": temperature,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": user}]}]}
        for attempt in range(3):
            try:
                t0 = time.time()
                r = await client.post(ENDPOINT, json=body, timeout=600.0)
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"]
                metrics["ok"] += 1
                return {"slug": row["slug"], "caption": txt,
                        "word_count": len(txt.split()),
                        "latency_s": time.time() - t0,
                        "model": MODEL, "prompt_tag": "v12", "error": ""}
            except Exception as e:
                if attempt == 2:
                    metrics["fail"] += 1
                    return {"slug": row["slug"], "caption": "", "word_count": 0,
                            "latency_s": 0.0, "model": MODEL, "prompt_tag": "v12",
                            "error": f"{type(e).__name__}: {str(e)[:200]}"}
                await asyncio.sleep(5 * (2 ** attempt))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", default="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts/grapher_charts_standalone.parquet")
    ap.add_argument("--out-parquet", default="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/owid___charts/grapher_charts_standalone_captioned.parquet")
    ap.add_argument("--prompt-file", default="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/owid/CAPTION_PROMPT_V12_OWID_PRODUCTION.txt")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--flush-every", type=int, default=500)
    args = ap.parse_args()

    system = Path(args.prompt_file).read_text().strip()
    df = pl.read_parquet(args.in_parquet)
    print(f"input: {len(df)} rows", flush=True)

    # Resume: skip slugs already in output
    done: set[str] = set()
    if Path(args.out_parquet).exists():
        prev = pl.read_parquet(args.out_parquet, columns=["slug"])
        done = set(prev["slug"].to_list())
        print(f"resume: {len(done)} already captioned", flush=True)

    rows = [r for r in df.iter_rows(named=True) if r["slug"] not in done]
    print(f"to caption: {len(rows)}", flush=True)
    if not rows: return

    sem = asyncio.Semaphore(args.concurrency)
    metrics = {"ok": 0, "fail": 0}
    buf: list[dict] = []
    t0 = time.time()

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency * 2)) as client:
        tasks = [asyncio.create_task(caption_one(client, sem, system, r, metrics)) for r in rows]
        for i, coro in enumerate(asyncio.as_completed(tasks), start=1):
            rec = await coro
            buf.append(rec)
            if i % 50 == 0:
                dt = time.time() - t0
                print(f"[{i:>5}/{len(rows)}] ok={metrics['ok']} fail={metrics['fail']} rate={i/max(dt,1):.1f}/s", flush=True)
            if len(buf) >= args.flush_every:
                _append(args.out_parquet, buf); buf.clear()
    if buf: _append(args.out_parquet, buf)

    dt = time.time() - t0
    print(f"DONE {dt/60:.1f} min: {metrics}", flush=True)


def _append(path, rows):
    new = pl.DataFrame(rows)
    p = Path(path)
    if p.exists():
        old = pl.read_parquet(p)
        combined = pl.concat([old, new], how="diagonal_relaxed")
    else:
        combined = new
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(p, compression="zstd")
    print(f"  flushed {len(rows)} -> {p} (total {len(combined)})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
