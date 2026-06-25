"""Extract a stratified sample of the perfect-OCR Smithsonian subset.

Outputs paired (jpg, txt) files for browsing in Yazi:

  perfect_samples/
    01_grade5_handwriting_<id>.jpg
    01_grade5_handwriting_<id>.txt
    ...

The .txt contains: id, dimensions, gate decision, ver grade, ver issues,
tag distribution, and the full OCR transcription.
"""
import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir",  default="/capstor/scratch/cscs/xyixuan/recon/smithsonian_ocr_perfect")
    ap.add_argument("--output-dir", default="/iopsstor/scratch/cscs/xyixuan/apertus/multimodal-data/01-dataset-download/image/smithsonian/ocr_pass/perfect_samples")
    ap.add_argument("--per-bucket", type=int, default=2,
                    help="how many samples to extract per (grade, dominant_tag) bucket")
    ap.add_argument("--min-chars", type=int, default=0,
                    help="minimum ocr_chars to qualify (filters short rows)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stream over shards, bucket by (grade, dominant_tag), keep first N per bucket
    buckets = defaultdict(list)
    shards = sorted(Path(args.input_dir).glob("*.parquet"))
    print(f"scanning {len(shards)} shards…")
    for s in shards:
        for r in pq.read_table(s).to_pylist():
            if r["ocr_chars"] < args.min_chars:
                continue
            try:
                td = json.loads(r["ocr_tags_json"]) if r["ocr_tags_json"] else {}
            except Exception:
                td = {}
            dom_tag = max(td.items(), key=lambda kv: kv[1])[0] if td else "untagged"
            key = (r["ver_grade"], dom_tag)
            if len(buckets[key]) < args.per_bucket:
                buckets[key].append(r)
        if all(len(v) >= args.per_bucket for v in buckets.values()) and len(buckets) >= 12:
            # Have enough variety; stop early
            break

    print(f"\n{len(buckets)} (grade, tag) buckets sampled:")
    for k, rows in sorted(buckets.items()):
        print(f"  grade={k[0]} tag={k[1]:<20s} n={len(rows)}")

    # Order: grade desc, tag alpha
    ordered = []
    for (grade, tag), rows in sorted(buckets.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        ordered.extend(rows[:args.per_bucket])

    print(f"\nwriting {len(ordered)} samples to {out_dir}")
    for i, r in enumerate(ordered):
        sid = r["id"].split("___")[-1][:40]  # short id for filename
        try:
            td = json.loads(r["ocr_tags_json"]) if r["ocr_tags_json"] else {}
        except Exception:
            td = {}
        dom_tag = max(td.items(), key=lambda kv: kv[1])[0] if td else "untagged"
        stem = f"{i:02d}_grade{r['ver_grade']}_{dom_tag}_{sid}"

        # Write image
        try:
            img = Image.open(io.BytesIO(r["image"]))
            (out_dir / f"{stem}.jpg").write_bytes(r["image"])
            w, h = img.size
        except Exception as e:
            (out_dir / f"{stem}.jpg").write_bytes(r["image"])
            w, h = "?", "?"

        # Write companion .txt
        txt = (out_dir / f"{stem}.txt")
        txt.write_text(
            f"id:           {r['id']}\n"
            f"dimensions:   {w}x{h}\n"
            f"ocr_chars:    {r['ocr_chars']:,}\n"
            f"tags:         {json.dumps(td)}\n"
            f"gate:         {r['gate_decision']} — {r['gate_reason']}\n"
            f"ver_grade:    {r['ver_grade']}/5  issues: {r['ver_issues']}\n"
            f"\n"
            f"=== OCR transcription ===\n"
            f"{r['ocr_text']}\n"
        )
    print("done.")
    print(f"\nbrowse with: yazi {out_dir}")


if __name__ == "__main__":
    main()
