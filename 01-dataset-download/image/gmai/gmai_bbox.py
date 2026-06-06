import json, re, glob, os
A="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___General-Medical-AI___GMAI-VL-5.5M/GMAI-VL-5.5M-OpenSource/annotations"
NUM=re.compile(r'-?\d+\.?\d*')
def coords_in(txt):
    # capture bracketed/paren coordinate groups
    return re.findall(r'[\[\(]\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*.*?[\]\)]', txt)
shown=0; allvals=[]; fmt=set(); hw=[]
for path in sorted(glob.glob(A+"/GMAI-MM-Percept-1.3M.jsonl")):  # Percept = grounding-heavy
    with open(path) as f:
        for ln in f:
            try: r=json.loads(ln)
            except: continue
            conv=r.get('conversations',[])
            blob=" ".join(c.get('value','') for c in conv)
            if 'box' not in blob.lower() and '<ref>' not in blob and 'bounding' not in blob.lower(): continue
            cs=coords_in(blob)
            if not cs: continue
            # collect numbers
            for c in cs:
                for v in NUM.findall(c):
                    try: allvals.append(float(v))
                    except: pass
            for c in cs[:2]: fmt.add(c.strip()[:60])
            h,w=r.get('height'),r.get('width'); 
            if h and w: hw.append((h,w))
            if shown<5:
                shown+=1
                im=r.get('image','')
                print(f"--- ex{shown}  h={r.get('height')} w={r.get('width')}  src={im.split('/')[4] if len(im.split('/'))>4 else '?'} ---")
                for c in conv: print(f"   {c.get('from')}: {c.get('value','')[:300]}")
            if len(allvals)>200000: break
import numpy as np
v=np.array(allvals)
print(f"\n=== coordinate value stats over {len(v):,} numbers ===")
print(f"  min={v.min():.2f}  max={v.max():.2f}  p50={np.percentile(v,50):.1f}  p99={np.percentile(v,99):.1f}")
print(f"  fraction in (0,1]={100*((v>0)&(v<=1)).mean():.1f}%   <=100={100*(v<=100).mean():.1f}%   <=1000={100*(v<=1000).mean():.1f}%   >1000={100*(v>1000).mean():.1f}%")
print(f"  sample format strings: {list(fmt)[:5]}")
if hw:
    import collections
    print(f"  image height/width seen (sample): {collections.Counter(hw).most_common(5)}")
