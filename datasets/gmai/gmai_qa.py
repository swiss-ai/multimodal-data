import json, glob, os
from collections import Counter, defaultdict
A="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___General-Medical-AI___GMAI-VL-5.5M/GMAI-VL-5.5M-OpenSource/annotations"
PERMISSIVE={"TotalSegmentator_v2","crc100k","PAD-UFES-20","AMOS_MR"}
# which files + task-paths + sample convos per permissive source
per_file_count=defaultdict(Counter)         # source -> file -> n
taskpath=defaultdict(Counter)               # source -> path[1]/path[2] -> n
samples=defaultdict(list)                    # source -> [(file, convo)]
qtypes=defaultdict(Counter)                  # source -> first-human-prompt (truncated) -> n
for path in sorted(glob.glob(A+"/GMAI-MM-*.jsonl")):
    fn=os.path.basename(path).split('-')[2]  # Caption/Instrunct/Percept
    with open(path) as f:
        for ln in f:
            try: r=json.loads(ln)
            except: continue
            im=r.get('image'); 
            if isinstance(im,list): im=im[0] if im else ''
            parts=(im or '').split('/')
            if len(parts)<5: continue
            src=parts[4]
            if src not in PERMISSIVE: continue
            per_file_count[src][fn]+=1
            taskpath[src][f"{parts[1]}/{parts[2]}"]+=1
            conv=r.get('conversations',[])
            if conv:
                hum=next((c['value'] for c in conv if c.get('from')=='human'),'')
                q=hum.replace('<image>','').strip()[:70]
                qtypes[src][q]+=1
                if len(samples[src])<3:
                    samples[src].append((fn,[(c.get('from'),c.get('value','')[:240]) for c in conv]))
if __name__=="__main__":
    for src in ["TotalSegmentator_v2","crc100k","PAD-UFES-20","AMOS_MR"]:
        print(f"\n{'='*70}\n{src}: {dict(per_file_count[src])}  tasks={dict(taskpath[src])}")
        print(f"  top question templates:")
        for q,c in qtypes[src].most_common(6): print(f"    [{c:>6,}] {q!r}")
        print(f"  sample conversations:")
        for fn,turns in samples[src][:2]:
            print(f"    --- from {fn} ---")
            for frm,val in turns: print(f"      {frm}: {val}")
