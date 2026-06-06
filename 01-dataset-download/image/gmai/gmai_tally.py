import json, re, glob, os
from collections import Counter
from multiprocessing import Pool
A="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___General-Medical-AI___GMAI-VL-5.5M/GMAI-VL-5.5M-OpenSource/annotations"
IMG_RE=re.compile(rb'"image"\s*:\s*"([^"]+)"')
def tally(path):
    src=Counter(); task=Counter(); n=0
    with open(path,'rb') as f:
        for ln in f:
            m=IMG_RE.search(ln)
            if not m: continue
            n+=1
            parts=m.group(1).decode('utf-8','ignore').split('/')
            if len(parts)>=5:
                src[parts[4]]+=1
                task[f"{parts[1]}/{parts[2]}"]+=1
    return os.path.basename(path), n, src, task
if __name__=="__main__":
    files=sorted(glob.glob(A+"/GMAI-MM-*.jsonl"))
    with Pool(3) as p: res=p.map(tally, files)
    allsrc=Counter(); alltask=Counter(); tot=0
    for name,n,src,task in res:
        print(f"{name}: {n:,} rows")
        tot+=n; allsrc.update(src); alltask.update(task)
    print(f"\nTOTAL multimodal rows: {tot:,}  | distinct source datasets: {len(allsrc)}")
    print("\n=== task-type breakdown ===")
    for k,v in alltask.most_common(): print(f"  {k:30s} {v:>10,}")
    print(f"\n=== ALL source datasets (rows) ===")
    for k,v in allsrc.most_common(): print(f"  {k:40s} {v:>10,}")
