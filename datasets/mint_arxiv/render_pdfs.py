#!/usr/bin/env python3
"""
Render downloaded arXiv PDFs to PNG images, organized to match JSONL paths.

Input:  data/pdfs/{arxiv_id}.pdf
Output: data/images/{tar_name}/{yymm}_{arxiv_id}_pdf/{page}.png

Crash-safe: renders to a .tmp directory first, then renames atomically.
Resume-safe: skips papers whose output directory already exists.
On restart, stale .tmp dirs are cleaned and re-rendered.

Usage:
    render_pdfs.py                   # render all, using all CPUs
    render_pdfs.py --workers N
    render_pdfs.py --dpi D           # default 150
    render_pdfs.py --test N          # first N PDFs only
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
PDF_DIR = BASE / "data" / "pdfs"
IMG_DIR = BASE / "data" / "images"
TAR_TO_IDS = BASE / "tar_to_ids.json"


def build_work_list(dpi: int) -> list[tuple]:
    with open(TAR_TO_IDS) as f:
        data = json.load(f)

    # arxiv_id -> (tar_name, yymm)
    id_map = {}
    for tar_name, ids in data.items():
        m = re.match(r"arXiv_pdf_(\d+)_", tar_name)
        yymm = m.group(1) if m else "0000"
        for aid in ids:
            id_map[aid] = (tar_name, yymm)

    work = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        arxiv_id = pdf_path.stem
        if arxiv_id not in id_map:
            continue
        tar_name, yymm = id_map[arxiv_id]
        out_dir = IMG_DIR / tar_name / f"{yymm}_{arxiv_id}_pdf"
        tmp_dir = out_dir.parent / f"{out_dir.name}.tmp"

        if out_dir.exists():
            continue  # already done

        work.append((str(pdf_path), str(out_dir), str(tmp_dir), dpi))

    return work


def render_one(args: tuple) -> tuple[str, int, str]:
    """Render one PDF atomically. Returns (name, n_pages, status)."""
    pdf_path_s, out_dir_s, tmp_dir_s, dpi = args
    pdf_path = Path(pdf_path_s)
    out_dir = Path(out_dir_s)
    tmp_dir = Path(tmp_dir_s)

    # Double-check in case another worker just finished it
    if out_dir.exists():
        return pdf_path.stem, 0, "skip"

    # Clean up stale tmp from a prior crash
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    try:
        import fitz  # import inside worker to avoid issues with fork+import

        doc = fitz.open(pdf_path)
        n = len(doc)
        mat = fitz.Matrix(dpi / 72, dpi / 72)

        tmp_dir.mkdir(parents=True, exist_ok=True)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            pix.save(tmp_dir / f"{i}.png")
        doc.close()

        # Atomic rename: only visible to others once fully complete
        os.rename(tmp_dir, out_dir)
        return pdf_path.stem, n, "ok"

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return pdf_path.stem, 0, f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--test", type=int, default=0)
    args = parser.parse_args()

    print("Building work list...", flush=True)
    work = build_work_list(args.dpi)

    already_done = sum(
        1
        for p in PDF_DIR.glob("*.pdf")
        if (IMG_DIR / "x" / "x").parent.parent  # just to avoid recomputing
        and True
    ) - len(work)
    # simpler count:
    total_pdfs = sum(1 for _ in PDF_DIR.glob("*.pdf"))
    already_done = total_pdfs - len(work)

    if args.test:
        work = work[: args.test]
        print(f"TEST MODE: rendering {len(work)} PDFs with {args.workers} workers")
    else:
        print(f"Total PDFs: {total_pdfs}  Already rendered: {already_done}  Remaining: {len(work)}")
        print(f"Workers: {args.workers}  DPI: {args.dpi}", flush=True)

    if not work:
        print("Nothing to do.")
        return

    t0 = time.monotonic()
    ok = skip = errors = total_pages = 0
    report_every = max(1, len(work) // 200)  # ~200 progress lines

    with mp.Pool(processes=args.workers) as pool:
        for i, (name, n_pages, status) in enumerate(pool.imap_unordered(render_one, work, chunksize=4)):
            if status == "ok":
                ok += 1
                total_pages += n_pages
            elif status == "skip":
                skip += 1
            else:
                errors += 1
                print(f"  WARN {name}: {status}", file=sys.stderr, flush=True)

            done = ok + skip
            if done > 0 and (i + 1) % report_every == 0:
                elapsed = time.monotonic() - t0
                rate = done / elapsed
                eta = (len(work) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1}/{len(work)}] ok={ok} skip={skip} err={errors} "
                    f"pages={total_pages}  {rate:.1f} PDF/s  ETA {eta / 60:.1f}min",
                    flush=True,
                )

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed / 60:.1f} min | ok={ok} skipped={skip} errors={errors} total_pages={total_pages}")


if __name__ == "__main__":
    main()
