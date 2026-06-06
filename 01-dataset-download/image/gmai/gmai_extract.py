import re, glob, os, json
from collections import Counter, defaultdict
A="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___General-Medical-AI___GMAI-VL-5.5M/GMAI-VL-5.5M-OpenSource/annotations"
OUT="/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/gmai_vl_permissive"
os.makedirs(OUT, exist_ok=True)
KEEP={"TotalSegmentator_v2","crc100k","PAD-UFES-20"}
IMG_RE=re.compile(rb'"image"\s*:\s*"([^"]+)"')
per=defaultdict(Counter); n_out=0; imgs=set()
fout=open(f"{OUT}/gmai_vl_permissive.jsonl","wb")
for path in sorted(glob.glob(A+"/GMAI-MM-*.jsonl")):
    fn=os.path.basename(path).split('-')[2]
    with open(path,'rb') as f:
        for ln in f:
            m=IMG_RE.search(ln)
            if not m: continue
            p=m.group(1).decode('utf-8','ignore').split('/')
            if len(p)<5 or p[4] not in KEEP: continue
            fout.write(ln if ln.endswith(b'\n') else ln+b'\n')
            per[p[4]][fn]+=1; n_out+=1; imgs.add(m.group(1).decode('utf-8','ignore'))
fout.close()
with open(f"{OUT}/image_paths.txt","w") as g:
    for ip in sorted(imgs): g.write(ip+"\n")
print(f"wrote {n_out:,} rows -> {OUT}/gmai_vl_permissive.jsonl")
print(f"unique images referenced: {len(imgs):,}")
for s in KEEP: print(f"  {s:22s} {dict(per[s])}  total={sum(per[s].values()):,}")
# top-level image dir prefixes (to know which zip chunks / subtrees we need)
pref=Counter('/'.join(ip.split('/')[:4]) for ip in imgs)
print("image subtree prefixes (depth4):")
for k,v in pref.most_common(): print(f"  {k}  -> {v:,} imgs")
