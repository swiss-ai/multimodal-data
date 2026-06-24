"""PubMedVision prep: (1) convert Instruct+Align JSON arrays -> one jsonl with
N <image> placeholders injected per row (N = #images, since the source has NONE
and rows are multi-image up to 29); (2) repack the 20 image zips -> 20 tars with
member names preserved (images/pmc_X.jpg == the jsonl image refs, no prefix fix).

Usage: python scripts/pubmedvision_prep.py [convert|repack|both]
"""
import sys, os, json, glob, zipfile, tarfile, io
from multiprocessing import Pool

SNAP = "/capstor/store/cscs/swissai/infra01/vision-datasets/hf_hub_cache/datasets--FreedomIntelligence--PubMedVision/snapshots/3c84e04b38bceb5341419b9a4f8ca37ba790cb84"
OUT = "/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/pubmedvision"
TARDIR = f"{OUT}/tars"
SUBSETS = ["PubMedVision_InstructionTuning_VQA", "PubMedVision_Alignment_VQA"]

def convert():
    os.makedirs(OUT, exist_ok=True)
    out = f"{OUT}/pubmedvision.jsonl"
    n_tot = 0
    per = {}
    with open(out, "w") as g:
        for j in SUBSETS:
            data = json.load(open(f"{SNAP}/{j}.json"))
            per[j] = len(data)
            for r in data:
                imgs = r.get("image", []) or []
                nimg = len(imgs)
                conv = r.get("conversations", [])
                # inject N placeholders into the FIRST human turn
                for t in conv:
                    if t.get("from") == "human":
                        t["value"] = ("<image>\n" * nimg) + t.get("value", "")
                        break
                g.write(json.dumps({
                    "image": imgs, "conversations": conv, "id": r.get("id"),
                    "modality": r.get("modality"), "body_part": r.get("body_part"),
                }) + "\n")
                n_tot += 1
            del data
    print(f"convert: wrote {n_tot:,} rows -> {out}  ({per})", flush=True)

def _repack(zp):
    n = int(os.path.basename(zp).replace("images_", "").replace(".zip", ""))
    out = f"{TARDIR}/pubmedvision_{n:02d}.tar"
    if os.path.exists(out):
        return f"skip {out}"
    cnt = 0
    with zipfile.ZipFile(zp) as z, tarfile.open(out + ".tmp", "w") as tf:
        for m in z.namelist():
            if m.endswith("/"):
                continue
            data = z.read(m)
            ti = tarfile.TarInfo(name=m); ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data)); cnt += 1
    os.rename(out + ".tmp", out)
    return f"{os.path.basename(out)}: {cnt:,} imgs"

def repack():
    os.makedirs(TARDIR, exist_ok=True)
    zips = sorted(glob.glob(f"{SNAP}/images_*.zip"))
    with Pool(min(20, len(zips))) as p:
        for r in p.imap_unordered(_repack, zips):
            print(r, flush=True)
    tot = sum(1 for _ in glob.glob(f"{TARDIR}/*.tar"))
    print(f"repack: {tot} tars in {TARDIR}", flush=True)

if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("convert", "both"): convert()
    if what in ("repack", "both"): repack()
