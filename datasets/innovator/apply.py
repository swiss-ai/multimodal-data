#!/usr/bin/env python3
"""Apply rewrites to data2 parquet files → OUTPUT_DIR (production drop-in)."""

import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA2_DIR = Path("/path/to/data/vision-datasets/raw/sft/hf___InnovatorLab___Innovator-VL-Instruct-46M/data2")
OUTPUT_DIR = Path(
    "/path/to/data/vision-datasets/processed/hf___InnovatorLab___Innovator-VL-Instruct-46M___restored_think_tokens2/data2"
)
REWRITES_DIR = Path("/tmp/restore_innovator/outputs/rewrites")
PATCHES_DIR = Path("/dev/shm/innovator_patches2")
MIN_ROUGE_L = 0.8
N_WORKERS = 32


def normalize_rewrite(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>\s+", "<think>", text)
    text = re.sub(r"\s+</think>", "</think>", text)
    return text


def is_valid_rewrite(text: str) -> bool:
    has_open = "<think>" in text
    has_close = "</think>" in text

    # No think tags → Qwen left it unchanged → valid
    if not has_open and not has_close:
        return True

    # Structural checks
    if not has_open or not has_close:  # one tag without the other
        return False
    if not text.startswith("<think>"):  # text before opening tag
        return False
    if text.count("<think>") > 1:  # multiple blocks
        return False

    inner, after = text[len("<think>") :].split("</think>", 1)

    if len(inner.strip()) < 20:  # empty or too-short think
        return False
    if not after.strip():  # no answer after closing tag
        return False
    if "<think>" in after or "</think>" in after:  # stray tags in answer
        return False

    return True


def stream_rewrites() -> None:
    """Stream rewrites one line at a time, writing per-file patch JSONL to PATCHES_DIR."""
    PATCHES_DIR.mkdir(exist_ok=True)
    handles = {}
    accepted = 0
    skipped = 0
    total = 0
    try:
        for jf in sorted(REWRITES_DIR.glob("*.jsonl")):
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    total += 1
                    rewritten = normalize_rewrite(rec["rewritten"])
                    if rec["rouge_l"] < MIN_ROUGE_L or "[truncated]" in rewritten or not is_valid_rewrite(rewritten):
                        skipped += 1
                        continue
                    fname = rec["file"]
                    if fname not in handles:
                        handles[fname] = open(PATCHES_DIR / fname, "w")
                    handles[fname].write(
                        json.dumps(
                            {
                                "row_idx": rec["row_idx"],
                                "turn_idx": rec["turn_idx"],
                                "rewritten": rewritten,
                            }
                        )
                        + "\n"
                    )
                    accepted += 1
    finally:
        for h in handles.values():
            h.close()
    print(f"Loaded {total} rewrites: {accepted} accepted, {skipped} failed filters (rouge_l/truncated/malformed)")


def patch_parquet(src: Path, dst: Path, patches: list) -> int:
    """Read src, apply patches, write dst. Returns number of rows patched."""
    table = pq.read_table(str(src))
    convs_col = table.column("conversations").to_pylist()

    patched = 0
    for rec in patches:
        row_idx, turn_idx, new_value = rec["row_idx"], rec["turn_idx"], rec["rewritten"]
        orig = convs_col[row_idx][turn_idx]["value"]
        if orig == new_value:
            continue  # Qwen left unchanged
        convs_col[row_idx][turn_idx] = dict(convs_col[row_idx][turn_idx])
        convs_col[row_idx][turn_idx]["value"] = new_value
        patched += 1

    if patched == 0:
        return 0

    conv_schema = table.schema.field("conversations").type
    new_convs = pa.array(convs_col, type=conv_schema)
    new_table = table.set_column(
        table.schema.get_field_index("conversations"),
        "conversations",
        new_convs,
    )
    pq.write_table(new_table, str(dst), compression="snappy")
    return patched


def process_file(args: tuple) -> tuple[int, bool]:
    src, dst = args
    if dst.exists():
        return 0, False
    patch_file = PATCHES_DIR / src.name
    if not patch_file.exists():
        os.link(src, dst)
        return 0, True
    patches = [json.loads(line) for line in patch_file.read_text().splitlines() if line.strip()]
    n = patch_parquet(src, dst, patches)
    if n == 0:
        os.link(src, dst)
        return 0, True
    return n, False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Streaming rewrites to /dev/shm ...", flush=True)
    stream_rewrites()

    all_files = sorted(DATA2_DIR.glob("*.parquet"))
    tasks = [(src, OUTPUT_DIR / src.name) for src in all_files]

    hardlinked = 0
    rewritten = 0
    total_patched_rows = 0

    print(
        f"Patching {len(all_files)} parquet files with {N_WORKERS} workers ...",
        flush=True,
    )
    with Pool(N_WORKERS) as pool:
        for n_patched, was_hardlinked in pool.imap_unordered(process_file, tasks):
            if was_hardlinked:
                hardlinked += 1
            else:
                rewritten += 1
                total_patched_rows += n_patched
            done = hardlinked + rewritten
            if done % 500 == 0:
                print(f"  {done}/{len(all_files)} files done", flush=True)

    print("\nDone:")
    print(f"  {hardlinked} files hardlinked (unmodified)")
    print(f"  {rewritten} files rewritten ({total_patched_rows} rows patched)")
    print(f"  Total in {OUTPUT_DIR}: {hardlinked + rewritten}")


if __name__ == "__main__":
    main()
