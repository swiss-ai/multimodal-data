#!/usr/bin/env python3
"""Reshard LLaVA-OneVision-2 spatial subsets (osd + roborefer_sim) into the
processed/sft jsonl_tar format.

Source (raw):
  spatial_instruct/osd_{choice,reasoning,template}.part_*.jsonl  -> osd
  spatial_instruct/roborefer_sim.part_*.jsonl                    -> roborefer_sim
  spatial/train_000{62..77}_of_00084.tar  hold images/positive/*  (osd images)
  spatial/train_000{78..80}_of_00084.tar  hold images/Simulator/* (roborefer images)

Output (processed/sft/<dataset>/):
  <dataset>.jsonl          normalized to {id, conversations:[{from,value}], image:[...]}
  images_part_NNN.tar      uncompressed image tars (one per source tarball)

Schema conversion: LOV2 ships {messages:[{role,content}], images, depth}.
  role user->human, assistant->gpt; content->value; images->image; depth dropped
  (depth maps reference an external RoboBrain2.5 path not present locally).

Only commercially-clean subsets are processed here (osd=OpenImages Apache-2.0,
roborefer_sim=Blender synthetic). ca1m (CC-BY-NC-ND) is intentionally excluded.

Bounded parallelism: one worker per source tarball (Pool over the 19 tarballs).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reshard_lov2_spatial")

RAW = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___mvp-lab___LLaVA-OneVision-2-Data")
PROC = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft")

DATASETS = {
    "osd": {
        "jsonl_glob": "osd_*.jsonl",
        "tar_indices": list(range(62, 78)),   # 62..77
        "prefix": "images/positive/",
        "out_dir": PROC / "hf___mvp-lab___LLaVA-OneVision-2-Data___osd",
    },
    "roborefer_sim": {
        "jsonl_glob": "roborefer_sim*.jsonl",
        "tar_indices": list(range(78, 81)),   # 78..80
        "prefix": "images/Simulator/",
        "out_dir": PROC / "hf___mvp-lab___LLaVA-OneVision-2-Data___roborefer_sim",
    },
}


def normalize_record(d: dict) -> dict:
    conv = []
    for m in d["messages"]:
        frm = "human" if m["role"] == "user" else "gpt"
        conv.append({"from": frm, "value": m["content"]})
    return {"id": d["id"], "conversations": conv, "image": d.get("images", [])}


def build_jsonl(name: str, cfg: dict) -> set[str]:
    """Normalize all JSONLs for a dataset, write merged output, return referenced image set."""
    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    referenced: set[str] = set()
    out_path = out_dir / f"{name}.jsonl"
    n = 0
    t0 = time.time()
    with open(out_path, "w") as out:
        for jf in sorted((RAW / "spatial_instruct").glob(cfg["jsonl_glob"])):
            with open(jf) as f:
                for line in f:
                    rec = normalize_record(json.loads(line))
                    referenced.update(rec["image"])
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
    logger.info("[%s] wrote %s: %d records, %d unique images (%.1fs)",
                name, out_path.name, n, len(referenced), time.time() - t0)
    return referenced


def extract_one_tar(args) -> tuple[str, int, int, float, str | None]:
    """Extract referenced members from one source tarball into one output tar."""
    name, src_idx, prefix, referenced, out_dir = args
    src = RAW / "spatial" / f"train_{src_idx:05d}_of_00084.tar"
    out_tar = Path(out_dir) / f"images_from_src{src_idx:05d}.tar"
    t0 = time.time()
    n_out = 0
    n_bytes = 0
    try:
        with tarfile.open(src, "r") as st, tarfile.open(out_tar, "w") as ot:
            for member in st:
                if not member.isfile():
                    continue
                if not member.name.startswith(prefix):
                    continue
                if member.name not in referenced:
                    continue
                f = st.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                ti = tarfile.TarInfo(name=member.name)
                ti.size = len(data)
                ti.mode = 0o644
                ti.mtime = int(t0)
                ot.addfile(ti, io.BytesIO(data))
                n_out += 1
                n_bytes += len(data)
    except Exception as e:
        return (f"src{src_idx:05d}", 0, 0, time.time() - t0, f"{type(e).__name__}: {e}")
    # remove empty output tars (boundary tarballs with no matching members)
    if n_out == 0:
        out_tar.unlink(missing_ok=True)
    return (f"src{src_idx:05d}", n_out, n_bytes, time.time() - t0, None)


def process_dataset(name: str, cfg: dict, workers: int) -> None:
    logger.info("=== processing %s ===", name)
    referenced = build_jsonl(name, cfg)

    tasks = [(name, ti, cfg["prefix"], referenced, str(cfg["out_dir"])) for ti in cfg["tar_indices"]]
    total_out = total_bytes = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(extract_one_tar, t): t[1] for t in tasks}
        for fut in as_completed(futs):
            tag, n_out, n_bytes, dur, err = fut.result()
            if err:
                logger.error("[%s] %s FAILED: %s", name, tag, err)
            else:
                logger.info("[%s] %s -> %d imgs, %.2f GB, %.1fs",
                            name, tag, n_out, n_bytes / 1e9, dur)
                total_out += n_out
                total_bytes += n_bytes
    missing = len(referenced) - total_out
    logger.info("[%s] DONE: %d imgs extracted (%.1f GB) in %.1fs; %d referenced not found",
                name, total_out, total_bytes / 1e9, time.time() - t0, missing)

    # provenance
    src_json = {
        "dataset_id": f"hf___mvp-lab___LLaVA-OneVision-2-Data___{name}",
        "source": "LLaVA-OneVision-2-Data spatial_instruct + spatial tarballs",
        "subset": name,
        "license": "apache-2.0" if name in ("osd", "roborefer_sim") else "unknown",
        "schema": "normalized to {id, conversations:[{from,value}], image:[...]}; depth dropped",
        "referenced_images": len(referenced),
        "extracted_images": total_out,
        "missing_images": missing,
        "notes": "Commercially-clean spatial-referring subset of LOV2. osd=OpenImages 2D (Apache-2.0); "
                 "roborefer_sim=Blender synthetic. ca1m (CC-BY-NC-ND) intentionally excluded. "
                 "Depth maps not available locally (external RoboBrain2.5 path).",
    }
    with open(cfg["out_dir"] / "_SOURCE.json", "w") as f:
        json.dump(src_json, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    args = ap.parse_args()
    for name in args.datasets:
        process_dataset(name, DATASETS[name], args.workers)
    logger.info("ALL DONE")


if __name__ == "__main__":
    main()
