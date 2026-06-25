"""Inline parallel filter runner — runs filter_finevision.filter_subset across
all 41 keep-list subsets using ProcessPoolExecutor. Skips subsets whose
download is incomplete (no .complete marker).

Output goes to /capstor/.../processed/sft/finevision/{subset}/train-*.parquet
(via filter_finevision.OUT_ROOT).
"""

import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_finevision import filter_subset, IN_ROOT

SUBSETS = [
    "DoclingMatix",
    # Synth{ChartNet,CodeNet,FormulaNet} dropped: IBM Granite Docling format
    # incompatible with BLIP-style grounding convention
    "cocoqa", "densefusion_1m", "localized_narratives", "nlvr2",
    # funsd dropped: 5% retention × 37% dup = no useful rows
    # synthdog dropped: 9.4% retention, OCR judge unreliable
    "multihiertt", "art", "wordart", "lnqa",
    "CoSyn_400k_chemical", "CoSyn_400k_circuit", "CoSyn_400k_document",
    "CoSyn_400k_graphic", "CoSyn_400k_math", "CoSyn_400k_music",
    "CoSyn_400k_nutrition", "CoSyn_400k_table",
    "mmevol", "aguvis-stage-1", "chinesememe", "memotion",
    # groundui dropped: redundant with molmopoint_guisyn
    "olmOCR-mix-0225-books", "olmOCR-mix-0225-documents",
    "tat_dqa", "svrd", "yesbut", "mmra", "latex_handwritten", "tal_ocr_eng",
    "coco_colors", "spot_the_diff", "latexformulas", "spark",
    "handwriting_forms", "wildvision",
]

WORKERS = 24  # 1 process per subset; some are large (densefusion 145 GB, lnqa 266 GB)


def _run(subset):
    if not (Path(IN_ROOT) / subset / ".complete").is_file():
        return {"subset": subset, "skipped": True, "reason": "download not complete"}
    t0 = time.time()
    try:
        stats = filter_subset(subset)
        stats["elapsed_s"] = time.time() - t0
        return stats
    except Exception as e:
        return {"subset": subset, "error": str(e), "elapsed_s": time.time() - t0}


if __name__ == "__main__":
    t0 = time.time()
    print(f"=== filtering {len(SUBSETS)} subsets with {WORKERS} workers ===\n", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_run, s): s for s in SUBSETS}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r.get("skipped"):
                print(f"[skip] {r['subset']}: {r['reason']}", flush=True)
            elif r.get("error"):
                print(f"[err]  {r['subset']}: {r['error']}", flush=True)
            else:
                pct = r["pct"]
                t = r["elapsed_s"]
                print(f"[ok]   {r['subset']}: {r['in']:,} -> {r['out']:,} ({pct:.1f}% kept) in {t:.0f}s", flush=True)
    print(f"\n=== done in {time.time()-t0:.0f}s ===")
    ok = [r for r in results if not r.get("skipped") and not r.get("error")]
    tot_in = sum(r["in"] for r in ok)
    tot_out = sum(r["out"] for r in ok)
    if tot_in:
        print(f"aggregate: {tot_in:,} -> {tot_out:,} ({100*tot_out/tot_in:.1f}% kept)")
