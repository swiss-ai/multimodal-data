"""Assemble the final perfect Smithsonian OCR parquet.

Joins every QA signal we've produced into a single parquet ready for cooldown
tokenization. Filter rule: grade ≥ 4 AND ocr_chars ≤ 5000 (drops hallucinated
drift outliers per README caveat) AND not NO_TEXT.

Output schema:
    id, image (bytes), ocr_text, ocr_chars, ocr_tags_json,
    gate_decision, gate_reason, ver_grade, ver_issues
"""
import argparse
import json
from pathlib import Path
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/smithsonian/smithsonian_cleaned5")
    ap.add_argument("--ocr-dir",   default="/capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_full")
    ap.add_argument("--gate-dir",  default="/capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_gate_platinum")
    ap.add_argument("--ver-dir",   default="/capstor/scratch/cscs/xyixuan/recon/qwen_smithsonian_verified")
    ap.add_argument("--output",    default="/capstor/scratch/cscs/xyixuan/recon/smithsonian_ocr_perfect")
    ap.add_argument("--min-grade", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=5000)
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load gate decisions
    print("loading gate decisions…")
    gate = {}
    for s in sorted(Path(args.gate_dir).glob("*.parquet")):
        for r in pq.read_table(s, columns=["id","gate_decision","gate_reason"]).to_pylist():
            gate[r["id"]] = r
    print(f"  → {len(gate):,} gate rows")

    # 2) Load verifier grades
    print("loading verifier grades…")
    ver = {}
    for s in sorted(Path(args.ver_dir).glob("*.parquet")):
        for r in pq.read_table(s, columns=["id","ver_grade","ver_issues"]).to_pylist():
            ver[r["id"]] = r
    print(f"  → {len(ver):,} verifier rows")

    # 3) Load OCR text (only for rows that pass gate KEEP + verifier grade ≥ min)
    print("filtering and joining…")
    keep_ids = set()
    for sid, g in gate.items():
        if g["gate_decision"] != "KEEP": continue
        v = ver.get(sid)
        if not v: continue
        if v["ver_grade"] < args.min_grade: continue
        keep_ids.add(sid)
    print(f"  → {len(keep_ids):,} ids pass gate-KEEP + grade ≥ {args.min_grade}")

    ocr = {}
    drop_chars = drop_no_text = 0
    for s in sorted(Path(args.ocr_dir).glob("*.parquet")):
        for r in pq.read_table(s, columns=["id","ocr_text","ocr_chars","ocr_no_text","ocr_tags_json"]).to_pylist():
            if r["id"] not in keep_ids: continue
            if r["ocr_no_text"]:
                drop_no_text += 1; continue
            if r["ocr_chars"] > args.max_chars:
                drop_chars += 1; continue
            ocr[r["id"]] = r
    print(f"  dropped {drop_no_text:,} NO_TEXT, {drop_chars:,} chars > {args.max_chars}")
    print(f"  → {len(ocr):,} final rows after heuristic post-filter")

    # 4) Read source images and assemble
    print("reading source images and writing output shards…")
    n_written = 0
    grade_dist = Counter()
    tag_dist = Counter()
    for s in sorted(Path(args.image_dir).glob("*.parquet")):
        out_path = out_dir / s.name
        if out_path.exists():
            print(f"[skip] {s.name}")
            continue
        rows = []
        for r in pq.read_table(s, columns=["id","image"]).to_pylist():
            sid = r["id"]
            if sid not in ocr: continue
            o = ocr[sid]
            v = ver[sid]
            g = gate[sid]
            rows.append({
                "id": sid,
                "image": r["image"],
                "ocr_text": o["ocr_text"],
                "ocr_chars": o["ocr_chars"],
                "ocr_tags_json": o["ocr_tags_json"],
                "gate_decision": g["gate_decision"],
                "gate_reason": g["gate_reason"],
                "ver_grade": v["ver_grade"],
                "ver_issues": v["ver_issues"],
            })
            grade_dist[v["ver_grade"]] += 1
            try:
                td = json.loads(o["ocr_tags_json"]) if o["ocr_tags_json"] else {}
                for k,c in td.items():
                    tag_dist[k] += 1
                    break  # count dominant tag once per row (first key in dict ≈ dominant)
            except Exception:
                pass
        if not rows:
            continue
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("image", pa.binary()),
            pa.field("ocr_text", pa.string()),
            pa.field("ocr_chars", pa.int64()),
            pa.field("ocr_tags_json", pa.string()),
            pa.field("gate_decision", pa.string()),
            pa.field("gate_reason", pa.string()),
            pa.field("ver_grade", pa.int32()),
            pa.field("ver_issues", pa.string()),
        ])
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_path, compression="zstd")
        n_written += len(rows)
        print(f"  [{s.name}] wrote {len(rows):,} rows (cumulative {n_written:,})")
    print(f"\n=== DONE: {n_written:,} rows written to {out_dir} ===")
    print(f"grade distribution: {dict(sorted(grade_dist.items()))}")
    print(f"tag distribution (top 8): {dict(tag_dist.most_common(8))}")


if __name__ == "__main__":
    main()
