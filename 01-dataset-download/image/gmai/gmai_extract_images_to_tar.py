"""Stream the 163,308 needed GMAI-permissive images out of the 14 OpenSource
zip chunks directly into uncompressed tar shards (no loose files on lustre).

Naming: the JSONL "image" field is  images/<X>  while zip members are  <X>
(no leading 'images/'). So jsonl_path == 'images/' + zip_member. We match on the
stripped form and write each tar member named as the FULL jsonl path, so the
jsonl_tar loader finds it with no prefix-strip needed.

Run after the zip download completes:
  python scripts/gmai_extract_images_to_tar.py
"""
import os, glob, zipfile, tarfile, io

BASE = "/capstor/store/cscs/swissai/infra01/vision-datasets"
ZIPS = sorted(glob.glob(f"{BASE}/raw/sft/hf___General-Medical-AI___GMAI-VL-5.5M/GMAI-VL-5.5M-OpenSource/zips/*.zip"))
OUT  = f"{BASE}/processed/sft/gmai_vl_permissive"
TARDIR = f"{OUT}/tars"
MANIFEST = f"{OUT}/image_paths.txt"
PREFIX = "images/"
SHARD_IMAGES = 50_000          # ~4 tar shards for 163k images

def main():
    needed = set(l.strip() for l in open(MANIFEST) if l.strip())
    # map zip-member form (no 'images/' prefix) -> full jsonl path
    stripped = {p[len(PREFIX):]: p for p in needed if p.startswith(PREFIX)}
    print(f"needed images: {len(needed):,} | with-prefix-mapped: {len(stripped):,} | zip chunks: {len(ZIPS)}")
    if not ZIPS:
        print("ERROR: no zip chunks found"); return 1
    os.makedirs(TARDIR, exist_ok=True)

    # verify on first zip
    with zipfile.ZipFile(ZIPS[0]) as z:
        members = [m for m in z.namelist() if not m.endswith('/')]
        overlap = sum(1 for m in members if m in stripped)
    print(f"first zip: {len(members):,} members | overlap-with-needed(stripped)={overlap:,}")
    if overlap == 0:
        print("STOP: still zero overlap — naming assumption wrong.");
        print("  sample members:", members[:3]); print("  sample stripped-needed:", list(stripped)[:3])
        return 2

    found = set(); shard_idx = 0; in_shard = 0
    def open_shard(i):
        p = f"{TARDIR}/gmai_vl_permissive_{i:03d}.tar"; return tarfile.open(p, "w"), p
    tf, cur = open_shard(shard_idx); print(f"writing -> {cur}")
    for zp in ZIPS:
        with zipfile.ZipFile(zp) as z:
            for m in z.namelist():
                if m.endswith('/') or m not in stripped or m in found:
                    continue
                data = z.read(m)
                ti = tarfile.TarInfo(name=stripped[m])     # write FULL jsonl path as member name
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
                found.add(m); in_shard += 1
                if in_shard >= SHARD_IMAGES:
                    tf.close(); shard_idx += 1; in_shard = 0
                    tf, cur = open_shard(shard_idx); print(f"  rotated -> {cur} ({len(found):,} so far)")
        print(f"  done {os.path.basename(zp)}: cumulative {len(found):,}/{len(stripped):,}")
    tf.close()

    missing = set(stripped) - found
    print(f"\nEXTRACTED {len(found):,}/{len(stripped):,} into {shard_idx+1} tar(s) at {TARDIR}")
    if missing:
        print(f"WARNING: {len(missing):,} needed images NOT found. Samples: {list(sorted(missing))[:5]}")
        with open(f"{OUT}/missing_images.txt","w") as g:
            for m in sorted(missing): g.write(PREFIX+m+"\n")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
