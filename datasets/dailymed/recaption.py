"""
DailyMed SPL per-image recaptioning with Gemma-4-31B-IT via vLLM.

Each image in a drug document gets its own grounded caption built from:
  - The image itself
  - The document header (drug name, manufacturer, active ingredients)
  - The markdown section where [[IMAGE: ...]] anchor appears

Output: one JSONL row per image.
  {"id", "doc_id", "image_index", "image_name", "caption"}
  id = "{doc_id}/{image_name}"

Environment variables (all optional with sensible defaults):
    TASK_ID          int   – this worker's 0-based index  (default 0)
    TASK_COUNT       int   – total number of workers       (default 1)
    MODEL_PATH       str   – path or HF repo for the model
    MODEL_CACHE_ROOT str   – parent dir of HF model cache
    GPU_MEMORY_UTIL  float – fraction of GPU memory to use (default 0.88)
    MAX_TOKENS       int   – max output tokens             (default 4096)
    BATCH_SIZE       int   – inference batch size          (default 8)
    TEST_MODE        1     – process only pinned sample images + a few others
    TEST_TOTAL_TASKS int   – total tasks in TEST_MODE      (default 10)
    INPUT_JSONL      str   – pre-dumped JSONL from dump_input.py
    DST_DIR          str   – output directory for caption JSONL files

Run (interactive test, 1 GPU):
    TEST_MODE=1 TASK_ID=0 TASK_COUNT=1 CUDA_VISIBLE_DEVICES=0 \\
    /path/to/dailymed_spl/.venv/bin/python recaption.py

Run (full, via slurm):
    TASK_ID=<n> TASK_COUNT=<total> /path/to/.venv/bin/python recaption.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TASK_ID = int(os.environ.get("TASK_ID", "0"))
TASK_COUNT = int(os.environ.get("TASK_COUNT", "1"))
TEST_MODE = os.environ.get("TEST_MODE", "") == "1"
TEST_TOTAL_TASKS = int(os.environ.get("TEST_TOTAL_TASKS", "10"))

DST_DIR = os.environ.get(
    "DST_DIR",
    "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_caption",
)
INPUT_JSONL = os.environ.get(
    "INPUT_JSONL",
    "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_caption/_input.jsonl",
)

_DEFAULT_CACHE = "/tmp/models"
_MODEL_SNAPSHOT = (
    Path(_DEFAULT_CACHE) / "models--google--gemma-4-31B-it" / "snapshots" / "439edf5652646a0d1bd8b46bfdc1d3645761a445"
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(_MODEL_SNAPSHOT) if _MODEL_SNAPSHOT.exists() else "google/gemma-4-31B-it",
)
MODEL_CACHE_ROOT = os.environ.get("MODEL_CACHE_ROOT", _DEFAULT_CACHE)

GPU_MEMORY_UTIL = float(os.environ.get("GPU_MEMORY_UTIL", "0.88"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

# 1 image (280 soft tokens) + header + section context fits easily in 8192.
MAX_MODEL_LEN = 16384

# Always include images from these docs in TEST_MODE.
PINNED_IDS = [
    "human_rx_part1/20210617_74bead6e-b746-4134-817f-0c053ef9e451",  # Enbrel, 200 images
]
# How many images to take from the pinned doc in TEST_MODE.
TEST_PINNED_IMAGES = 5


def _configure_worker_dirs() -> None:
    root = Path(f"/tmp/{os.environ.get('USER', 'user')}/spl-recap-{TASK_ID}")
    for key, sub in {
        "HF_HOME": "huggingface",
        "HUGGINGFACE_HUB_CACHE": "hub",
        "XDG_CACHE_HOME": "xdg-cache",
        "TRITON_CACHE_DIR": "triton",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
        "VLLM_CACHE_ROOT": "vllm",
        "VLLM_RPC_BASE_PATH": "vllm-rpc",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }.items():
        path = root / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


# Lines of context to take before and after the image anchor.
CONTEXT_WINDOW_LINES = 40


def extract_image_context(markdown: str, image_name: str) -> tuple[str, str]:
    """
    Returns (header, section) where:
      header  – lines before the first ## heading (drug name, manufacturer, etc.)
      section – up to CONTEXT_WINDOW_LINES lines before and after the image anchor,
                with the anchor line itself removed; falls back to first section.
    """
    lines = markdown.splitlines()

    # Header: everything before the first ## heading
    header_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            break
        header_lines.append(line)
    header = "\n".join(header_lines).strip()

    # Find the anchor line index
    anchor_suffix = f"| {image_name}]]"
    anchor_idx: int | None = None
    for i, line in enumerate(lines):
        if anchor_suffix in line:
            anchor_idx = i
            break

    if anchor_idx is not None:
        start = max(0, anchor_idx - CONTEXT_WINDOW_LINES)
        end = min(len(lines), anchor_idx + CONTEXT_WINDOW_LINES + 1)
        context_lines = [ln for ln in lines[start:end] if anchor_suffix not in ln]
        section = "\n".join(context_lines).strip()
    else:
        # No anchor: use the first ## section as fallback
        in_section = False
        section_lines: list[str] = []
        for line in lines:
            if line.startswith("## "):
                if in_section:
                    break
                in_section = True
            if in_section:
                section_lines.append(line)
        section = "\n".join(section_lines).strip()

    return header, section


def doc_to_tasks(row: dict) -> list[dict]:
    images = row.get("images") or []
    markdown = row.get("markdown") or ""
    doc_id = row["id"]

    tasks = []
    for i, img in enumerate(images):
        image_name = img["name"]
        header, section = extract_image_context(markdown, image_name)
        tasks.append(
            {
                "id": f"{doc_id}/{image_name}",
                "doc_id": doc_id,
                "image_index": i,
                "image_name": image_name,
                "image": img,
                "header": header,
                "section": section,
            }
        )
    return tasks


SYSTEM_PROMPT = """\
You are a medical writer producing interleaved pharmaceutical documents.

You receive one image from an FDA drug label together with the document header and \
the label section where the image appears.

Write a rich, information-dense passage that will be inserted at the image position \
in the document. It must read naturally as inline prose.

Rules:
- Open with a noun phrase that names the subject of the image \
  (e.g. "The bar graph above", "The SureClick autoinjector", "A photograph of the packaging"). \
  Do not start with a verb, a gerund, or "This image".
- Continue in flowing prose — no bullet points, no headers, no lists.
- Weave in relevant details from the surrounding label text: drug name, active ingredients, \
  dosing, indications, warnings, administration steps, storage conditions — whatever is \
  pertinent to what the image depicts.
- Be thorough. Do not artificially shorten the description.
"""


def build_messages(task: dict) -> list[dict]:
    img = task["image"]
    media_type = img.get("media_type", "image/jpeg")

    content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{img['b64']}"},
        }
    ]

    parts = []
    if task["header"]:
        parts.append(f"Document header:\n{task['header']}")
    if task["section"]:
        parts.append(f"Label section context:\n{task['section']}")
    parts.append("Write a rich inline passage describing the image using all available context above.")

    content.append({"type": "text", "text": "\n\n".join(parts)})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def load_all_rows() -> dict[str, dict]:
    assert Path(INPUT_JSONL).exists(), f"Input JSONL not found: {INPUT_JSONL}\nRun dump_input.py first."
    by_id: dict[str, dict] = {}
    with open(INPUT_JSONL, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_id[r["id"]] = r
    return by_id


def load_assigned_tasks() -> list[dict]:
    by_id = load_all_rows()
    all_ids = list(by_id.keys())
    total = len(all_ids)

    chunk = (total + TASK_COUNT - 1) // TASK_COUNT
    start = TASK_ID * chunk
    end = min(start + chunk, total)

    if TEST_MODE:
        pinned_tasks: list[dict] = []
        for pid in PINNED_IDS:
            if pid in by_id:
                pinned_tasks.extend(doc_to_tasks(by_id[pid])[:TEST_PINNED_IMAGES])

        other_tasks: list[dict] = []
        for rid in all_ids[start:end]:
            if rid not in set(PINNED_IDS):
                other_tasks.extend(doc_to_tasks(by_id[rid]))
                if len(pinned_tasks) + len(other_tasks) >= TEST_TOTAL_TASKS:
                    break

        tasks = pinned_tasks + other_tasks[: max(0, TEST_TOTAL_TASKS - len(pinned_tasks))]
        print(
            f"[worker {TASK_ID}/{TASK_COUNT}] TEST_MODE: {len(tasks)} tasks "
            f"(pinned={len(pinned_tasks)}, others={len(tasks) - len(pinned_tasks)})",
            flush=True,
        )
    else:
        rows = [by_id[rid] for rid in all_ids[start:end]]
        tasks = []
        for row in rows:
            tasks.extend(doc_to_tasks(row))
        print(
            f"[worker {TASK_ID}/{TASK_COUNT}] assigned {len(rows)} docs → "
            f"{len(tasks)} image tasks (rows {start}–{end - 1} of {total})",
            flush=True,
        )

    return tasks


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with open(out_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


def main() -> None:
    _configure_worker_dirs()

    Path(DST_DIR).mkdir(parents=True, exist_ok=True)
    out_path = Path(DST_DIR) / f"task_{TASK_ID:04d}.jsonl"

    all_tasks = load_assigned_tasks()
    done_ids = load_done_ids(out_path)
    pending = [t for t in all_tasks if t["id"] not in done_ids]
    print(
        f"[worker {TASK_ID}] {len(done_ids)} already done, {len(pending)} pending",
        flush=True,
    )
    if not pending:
        print(f"[worker {TASK_ID}] nothing to do", flush=True)
        return

    from vllm import LLM, SamplingParams

    print(f"[worker {TASK_ID}] loading model {MODEL_PATH}", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        download_dir=MODEL_CACHE_ROOT,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=BATCH_SIZE,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(
        temperature=0.2,
        top_p=0.90,
        max_tokens=MAX_TOKENS,
    )
    print(f"[worker {TASK_ID}] model loaded", flush=True)

    n_written = len(done_ids)

    with open(out_path, "a", encoding="utf-8") as out_fh:
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start : batch_start + BATCH_SIZE]
            conversations = [build_messages(t) for t in batch]
            try:
                outputs = llm.chat(
                    conversations,
                    sampling_params=sampling,
                    use_tqdm=False,
                )
                for task, out in zip(batch, outputs):
                    caption = out.outputs[0].text.strip()
                    record = {
                        "id": task["id"],
                        "doc_id": task["doc_id"],
                        "image_index": task["image_index"],
                        "image_name": task["image_name"],
                        "caption": caption,
                    }
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1
            except Exception as e:
                print(f"[worker {TASK_ID}] batch {batch_start} error: {e}", flush=True)
                for task in batch:
                    record = {
                        "id": task["id"],
                        "doc_id": task["doc_id"],
                        "image_index": task["image_index"],
                        "image_name": task["image_name"],
                        "caption": "",
                    }
                    out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1
            out_fh.flush()

            print(
                f"[worker {TASK_ID}] {n_written}/{len(all_tasks)} tasks done "
                f"(batch {batch_start}–{batch_start + len(batch) - 1})",
                flush=True,
            )

    print(f"[worker {TASK_ID}] done → {out_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
