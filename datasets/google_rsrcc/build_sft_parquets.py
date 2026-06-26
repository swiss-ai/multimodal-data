"""Convert google/RSRCC processed parquets to multi-image SFT format.

Source schema (per row):
    sample_id, source_sample_id, before (Binary), after (Binary),
    question (str), answer (str), reasoning (str), metadata (struct)

Output schema (multi-image SFT, matches pixmo-style chat layout):
    sample_id (str)
    images   (list[Binary])  ← [before_bytes, after_bytes]
    messages (list[struct{role, content}])
        [
          {"role": "user",      "content": "<image>\\n<image>\\n{question}"},
          {"role": "assistant", "content": "<think>\\n{reasoning}\\n</think>\\n{answer}"}
        ]

Two `<image>` markers in the first user turn so the SFT renderer splices the
``before`` and ``after`` images in order.

CoT is wrapped in ``<think>...</think>`` so the merge step's
``--strip-thinking`` filter can produce a no-CoT variant from the same tokenized
output (mirrors the ``mmfinereason_1_8m`` / ``mmfinereason_1_8m_no_cot`` pattern).

Writes via pyarrow (avoids the polars large-binary write bug).
"""
from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


def build_messages_struct() -> pa.StructType:
    return pa.struct([
        pa.field("role", pa.string()),
        pa.field("content", pa.string()),
    ])


def convert_one(src_path: str, dst_dir: str) -> tuple[str, int, str]:
    fname = os.path.basename(src_path)
    out_path = os.path.join(dst_dir, fname)
    if os.path.exists(out_path):
        n = len(pl.read_parquet(out_path, columns=["sample_id"]))
        return fname, n, "skipped"
    t0 = time.time()
    df = pl.read_parquet(
        src_path,
        columns=["sample_id", "before", "after", "question", "answer", "reasoning"],
    )
    n = len(df)
    if n == 0:
        return fname, 0, "empty"

    sample_ids = df["sample_id"].to_list()
    befores = df["before"].to_list()
    afters = df["after"].to_list()
    questions = df["question"].to_list()
    answers = df["answer"].to_list()
    reasonings = df["reasoning"].to_list()

    images: list[list[bytes]] = []
    messages: list[list[dict]] = []

    for q, a, r, b, af in zip(questions, answers, reasonings, befores, afters):
        images.append([b, af])
        user_content = f"<image>\n<image>\n{q}"
        if r:
            assistant_content = f"<think>\n{r}\n</think>\n{a}"
        else:
            assistant_content = a
        messages.append([
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ])

    images_arr = pa.array(images, type=pa.list_(pa.binary()))
    messages_arr = pa.array(messages, type=pa.list_(build_messages_struct()))
    sample_id_arr = pa.array(sample_ids, type=pa.string())

    out_tbl = pa.table({
        "sample_id": sample_id_arr,
        "images": images_arr,
        "messages": messages_arr,
    })
    pq.write_table(out_tbl, out_path, compression="zstd")
    return fname, n, f"written in {time.time()-t0:.1f}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/hf___google___RSRCC")
    ap.add_argument("--dst", default="/capstor/scratch/cscs/xyixuan/vision-datasets-processed/google_rsrcc_sft")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    Path(args.dst).mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(f"{args.src}/task*.parquet"))
    print(f"converting {len(shards)} shards -> {args.dst}", flush=True)

    total = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(convert_one, s, args.dst) for s in shards]
        done = 0
        for f in as_completed(futs):
            fname, n, status = f.result()
            total += n
            done += 1
            if done % 16 == 0 or done == len(shards):
                el = (time.time() - t_start) / 60
                print(f"  {done}/{len(shards)}  rows={total:,}  elapsed={el:.1f}min", flush=True)

    import subprocess
    sz = subprocess.check_output(["du", "-sh", args.dst]).decode().split()[0]
    print(f"\nDONE: {total:,} rows, {sz}, elapsed {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
