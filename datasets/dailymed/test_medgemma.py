"""
Interactive test: use MedGemma-1.5-4b-it to generate contextual captions
that replace [[IMAGE: ...]] placeholders in SPL documents.

Run (inside uenv):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python test_medgemma.py
"""

import glob

import pyarrow.parquet as pq

MODEL = "/tmp/models/models--google--medgemma-1.5-4b-it/snapshots/91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b"
PARQUET_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_md"

SYSTEM = """\
You are a medical writer producing interleaved pharmaceutical documents.
You are given a drug label document and one image that appears at a marked position in it.
Write professional prose that describes the image and its pharmaceutical context.
Use the full surrounding text — ingredients, dosing, indications, warnings, instructions — \
to write a rich, information-dense description. The text will be inserted directly at the \
image position and must read naturally inline.
"""

# ── load a handful of representative docs ────────────────────────────────────


def load_docs(targets: list[str]) -> list[dict]:
    files = sorted(glob.glob(f"{PARQUET_DIR}/part-*.parquet"))
    results = {}
    for f in files:
        tbl = pq.read_table(f, columns=["id", "markdown", "images"]).to_pydict()
        for id_, md, imgs in zip(tbl["id"], tbl["markdown"], tbl["images"]):
            for t in targets:
                if t in id_ and t not in results:
                    results[t] = {"id": id_, "markdown": md or "", "images": imgs or []}
        if len(results) == len(targets):
            break
    return list(results.values())


TARGETS = [
    "0c01c369-eb55-444c-aa5a-5d963a063ccf",  # BTB Plus  (image at end)
    "74bead6e-b746-4134-817f-0c053ef9e451",  # Enbrel    (many images, long doc)
]

docs = load_docs(TARGETS)
print(f"Loaded {len(docs)} docs")
for d in docs:
    print(f"  {d['id']}  —  {len(d['images'])} images")

# ── build vLLM messages ───────────────────────────────────────────────────────

CONTEXT_LINES = 40
MAX_CONTEXT_CHARS = 20_000  # ~5k tokens; drop image if exceeded


def extract_context(markdown: str, image_name: str) -> str | None:
    """Returns header + windowed section around the image anchor, or None to drop."""
    lines = markdown.splitlines()
    header_lines = []
    for line in lines:
        if line.startswith("## "):
            break
        header_lines.append(line)
    header = "\n".join(header_lines).strip()[:800]

    anchor = f"| {image_name}]]"
    idx = next((i for i, line in enumerate(lines) if anchor in l), None)
    if idx is None:
        return None  # anchor not found; drop

    start = max(0, idx - CONTEXT_LINES)
    end = min(len(lines), idx + CONTEXT_LINES + 1)
    section = "\n".join(line for line in lines[start:end] if anchor not in l)

    context = f"{header}\n\n---\n\n{section}"
    if len(context) > MAX_CONTEXT_CHARS:
        return None  # too long; drop
    return context


def build_messages(doc: dict, img: dict) -> list[dict] | None:
    import base64

    context = extract_context(doc["markdown"], img["name"])
    if context is None:
        return None
    b64 = base64.b64encode(img["bytes"]).decode()
    media_type = img.get("media_type", "image/jpeg")

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
        {
            "type": "text",
            "text": f"Document context:\n\n{context}\n\nWrite a rich inline description of the image using all available context above.",
        },
    ]
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ── load model ────────────────────────────────────────────────────────────────

from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL,
    dtype="bfloat16",
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1},
)
sampling = SamplingParams(temperature=0.2, top_p=0.9, max_tokens=512)
print("Model loaded.\n")

# ── run ───────────────────────────────────────────────────────────────────────

for doc in docs:
    imgs = doc["images"]
    # test first image and, for Enbrel, also image index 5
    indices = [0] if len(imgs) == 1 else [0, min(5, len(imgs) - 1)]
    for idx in indices:
        img = imgs[idx]
        msgs = build_messages(doc, img)
        if msgs is None:
            print(f"=== {doc['id']}  image[{idx}]: {img['name']} — DROPPED (no anchor / context too long) ===\n")
            continue
        out = llm.chat([msgs], sampling_params=sampling, use_tqdm=False)
        caption = out[0].outputs[0].text.strip()

        print(f"=== {doc['id']}  image[{idx}]: {img['name']} ===")
        anchor = f"| {img['name']}]]"
        anchor_line = next(
            (line for line in doc["markdown"].splitlines() if anchor in l),
            "(placeholder not found)",
        )
        print(f"Anchor: {anchor_line}")
        print(f"Caption:\n{caption}")
        print()
