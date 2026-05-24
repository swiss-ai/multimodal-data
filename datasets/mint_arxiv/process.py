#!/usr/bin/env python3
"""
Download arxiv PDF tars from S3, render PDFs to PNG pages, pack into output tars.

Usage:
    process.py <tar_name> [tar_name ...]
    process.py --range <start> <end>   # process tar_to_ids.json[start:end]
    process.py --all                    # process all entries in tar_to_ids.json

Output: data/<tar_name>.tar containing:
    {YYMM}_{arxiv_id}_pdf/{page}.png
    matching the path structure used in JSONL metadata.
"""

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import fitz  # pymupdf
from tqdm import tqdm

DATA_DIR = Path(__file__).parent / "data"
TAR_TO_IDS = Path(__file__).parent / "tar_to_ids.json"
DPI = 150  # render resolution


def s3_download_proc(tar_name):
    s3_path = f"s3://arxiv/pdf/{tar_name}.tar"
    return subprocess.Popen(
        ["aws", "s3", "cp", s3_path, "-", "--request-payer", "requester"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def render_pdf(pdf_bytes):
    """Return list of PNG bytes (one per page)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise RuntimeError(f"fitz open failed: {e}") from e
    pages = []
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        pages.append(pix.tobytes("png"))
    return pages


def arxiv_id_from_member(name):
    """Extract arxiv ID from a tar member path like '1608.03650.pdf' or 'pdf/1608.03650.pdf'."""
    stem = Path(name).stem  # strip directory and extension
    # Handle old-style IDs: hep-th0101001 -> keep as-is
    # Handle new-style IDs: 1608.03650 -> keep as-is
    return stem


def process_tar(tar_name, needed_ids, data_dir):
    """Download and render one arxiv PDF tar. Returns (rendered, skipped, errors)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    out_tar_path = data_dir / f"{tar_name}.tar"
    tmp_path = data_dir / f"{tar_name}.tar.tmp"

    if out_tar_path.exists():
        print(f"[SKIP] {tar_name} already done")
        return 0, len(needed_ids), 0

    # Extract YYMM from tar name: arXiv_pdf_1608_010 -> 1608
    m = re.match(r"arXiv_pdf_(\d+)_", tar_name)
    yymm = m.group(1) if m else "0000"

    print(f"[START] {tar_name}: {len(needed_ids)} arxiv IDs needed", flush=True)
    proc = s3_download_proc(tar_name)

    rendered = 0
    errors = 0
    found_ids = set()

    try:
        with (
            tarfile.open(fileobj=proc.stdout, mode="r|") as in_tar,
            tarfile.open(tmp_path, "w") as out_tar,
        ):
            for member in in_tar:
                if not member.name.endswith(".pdf"):
                    continue

                arxiv_id = arxiv_id_from_member(member.name)
                if arxiv_id not in needed_ids:
                    continue

                found_ids.add(arxiv_id)
                fobj = in_tar.extractfile(member)
                if fobj is None:
                    continue
                pdf_bytes: bytes = fobj.read()  # type: ignore[union-attr]

                try:
                    pngs = render_pdf(pdf_bytes)
                except Exception as e:
                    print(f"  [ERR] render {arxiv_id}: {e}", flush=True)
                    errors += 1
                    continue

                pdf_dir = f"{yymm}_{arxiv_id}_pdf"
                for page_num, png_bytes in enumerate(pngs):
                    info = tarfile.TarInfo(name=f"{pdf_dir}/{page_num}.png")
                    info.size = len(png_bytes)
                    out_tar.addfile(info, io.BytesIO(png_bytes))

                rendered += 1

    except Exception as e:
        print(f"[ERR] {tar_name}: {e}", flush=True)
        tmp_path.unlink(missing_ok=True)
        proc.kill()
        return rendered, len(needed_ids) - rendered - errors, errors

    proc.wait()
    stderr_output = proc.stderr.read().decode(errors="replace")
    if proc.returncode != 0:
        print(f"[ERR] S3 download failed for {tar_name}: {stderr_output}", flush=True)
        tmp_path.unlink(missing_ok=True)
        return 0, len(needed_ids), 0

    tmp_path.rename(out_tar_path)
    missing = needed_ids - found_ids
    if missing:
        print(f"  [WARN] {tar_name}: {len(missing)} IDs not found in tar", flush=True)
    print(
        f"[DONE] {tar_name}: rendered={rendered} errors={errors} missing={len(missing)}",
        flush=True,
    )
    return rendered, len(missing), errors


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Process all tar files")
    group.add_argument("--range", nargs=2, type=int, metavar=("START", "END"))
    group.add_argument("tar_names", nargs="*", help="Specific tar names to process")
    args = parser.parse_args()

    with open(TAR_TO_IDS) as f:
        tar_to_ids = json.load(f)

    all_tar_names = sorted(tar_to_ids.keys())

    if args.all:
        targets = all_tar_names
    elif args.range:
        start, end = args.range
        targets = all_tar_names[start:end]
    elif args.tar_names:
        targets = args.tar_names
    else:
        parser.print_help()
        sys.exit(1)

    print(f"Processing {len(targets)} tar files...", flush=True)

    total_rendered = total_missing = total_errors = 0
    for tar_name in tqdm(targets, desc="tars"):
        needed = set(tar_to_ids.get(tar_name, []))
        r, m, e = process_tar(tar_name, needed, DATA_DIR)
        total_rendered += r
        total_missing += m
        total_errors += e

    print(f"\nSummary: rendered={total_rendered} missing={total_missing} errors={total_errors}")


if __name__ == "__main__":
    main()
