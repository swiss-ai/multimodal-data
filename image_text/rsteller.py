import json
import os

import webdataset as wds

SRC_DIR = os.getenv("RSTELLER_SRC_DIR", "/path/to/data")
OUT_DIR = os.getenv("RSTELLER_OUT_DIR", "/path/to/output")

src_tars = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".tar"))
src_urls = [os.path.join(SRC_DIR, f) for f in src_tars]

pattern = os.path.join(OUT_DIR, "part-%06d.tar")
os.makedirs(OUT_DIR, exist_ok=True)
sink = wds.ShardWriter(pattern, maxcount=20000)

dataset = wds.WebDataset(src_urls, shardshuffle=False)

for i, sample in enumerate(dataset):
    data = sample["json"]
    data = json.loads(data.decode("utf-8"))
    caption = next(a["text"] for a in data["annotations"] if a["task"] in (1, 2))
    assert caption is not None

    assert "<|img1|>" not in caption
    caption = "<|img1|>\n" + caption

    sink.write(
        {
            "__key__": f"{i:06d}_{sample['__key__']}",
            "img1.jpg": sample["jpg"],
            "txt": caption,
        }
    )

    if (i + 1) % 50000 == 0:
        print(f"  {i + 1} samples written")

sink.close()
print(f"Finished. {i + 1} samples written.")
