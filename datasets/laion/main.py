import os

from datasets import load_dataset

# fmt: off

os.environ.setdefault("HF_HOME", os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface")))
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))

# fmt: on

batch_size = 1000
output_filename = "all_urls.txt"

ds = load_dataset(
    "dclure/laion-aesthetics-12m-umap",
    split="train",
    streaming=True,
)
ds = ds.select_columns(["URL"])
print(ds)

with open(output_filename, "w", encoding="utf-8") as f:
    for bi, batch in enumerate(ds.iter(batch_size=batch_size)):
        valid_urls = [url for url in batch["URL"] if url is not None]
        f.write("\n".join(valid_urls) + "\n")
        if (bi + 1) % 100 == 0:
            print(f"Processed {bi + 1} batches...")
