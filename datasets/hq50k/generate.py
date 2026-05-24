#!/usr/bin/env python3
"""HQ-50K cooldown caption generation via remote OpenAI-compatible API.

Each SLURM task handles one tar shard (or a sub-range of images within a shard).
Checkpoints to JSONL so reruns safely resume from where they left off.

Usage (single shard, manual):
    TASK_ID=0 TASK_COUNT=6 .venv/bin/python generate.py

Usage (SLURM array):
    See run.slurm
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
import tarfile
from datetime import UTC, datetime
from hashlib import blake2b
from pathlib import Path

import pyarrow.ipc as ipc
from openai import OpenAI

# ── Paths ──────────────────────────────────────────────────────────────────────

HQ50K_ROOT = Path("/path/to/data/vision-datasets/raw/cooldown/hf___YangQiee___HQ-50K/downloaded")
DOCCI_ARROW = Path("/path/to/data/vision-datasets/raw/stage2/hf___google___docci/docci-train.arrow")
DOCCI_IMAGES_DIR = DOCCI_ARROW.parent / "images"

_default_output_dir = Path(__file__).parent / "artifacts" / "candidates"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", str(_default_output_dir)))

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8080/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-397B-A17B-KPhL")

N_DOCCI_EXAMPLES = 10
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 1024
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

PROMPT_VERSION = "hq50k_cooldown_docci10_v10"

PROMPT = (
    "Describe this image accurately and in detail. "
    "Write in plain, continuous prose — no markdown, no bullet points, no headers, "
    "no bold or italic text. "
    'Do not open with "The image shows", "In this photograph", or any phrase '
    "referencing the medium or camera. Do not comment on mood, atmosphere, or "
    "artistic intent. Describe only what is directly visible."
)

FEWSHOT_PROMPT = PROMPT
CAPTION_PROMPT = PROMPT

# ── Helpers ────────────────────────────────────────────────────────────────────


def to_data_url(image_bytes: bytes, media_type: str = "image/jpeg") -> str:
    return f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}"


def load_docci_index() -> list[dict]:
    """Load DOCCI captions + image bytes index (cached once per process)."""
    with ipc.open_stream(DOCCI_ARROW) as f:
        table = f.read_all()
    index = []
    for i in range(len(table)):
        example_id = table["example_id"][i].as_py()
        img_path = DOCCI_IMAGES_DIR / f"{example_id}.jpg"
        if img_path.exists():
            index.append(
                {
                    "img_path": img_path,
                    "caption": table["description"][i].as_py(),
                }
            )
    print(f"Loaded {len(index)} DOCCI examples", flush=True)
    return index


def list_shard_members(shard_path: Path) -> list[str]:
    """Return sorted list of .jpg member names inside a tar shard."""
    with tarfile.open(shard_path, "r") as tf:
        return sorted(m.name for m in tf.getmembers() if m.name.endswith(".jpg"))


def read_image_from_shard(shard_path: Path, member_name: str) -> bytes:
    with tarfile.open(shard_path, "r") as tf:
        fobj = tf.extractfile(member_name)
        if fobj is None:
            raise FileNotFoundError(f"{member_name} not found in {shard_path}")
        return fobj.read()


def stable_task_id(sample_id: str, task_count: int) -> int:
    """Deterministically assign a task id by hashing the sample key."""
    digest = blake2b(sample_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % task_count


def build_messages(
    target_image_bytes: bytes,
    docci_examples: list[dict],
) -> list[dict]:
    """Build few-shot chat messages: 10 DOCCI (user+assistant) then the target."""
    messages: list[dict] = []
    for ex in docci_examples:
        ex_bytes = ex["img_path"].read_bytes()
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": to_data_url(ex_bytes)}},
                    {"type": "text", "text": FEWSHOT_PROMPT},
                ],
            }
        )
        messages.append({"role": "assistant", "content": ex["caption"]})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": to_data_url(target_image_bytes)},
                },
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    )
    return messages


import re as _re

_ABSENCE_RE = _re.compile(
    r"(?<=[.!?])\s+(?:No |There (?:is|are) no |There (?:is|are) nothing )[^.!?]+[.!?]",
    _re.IGNORECASE,
)


def strip_absence_sentences(text: str) -> str:
    """Remove sentences that assert the absence of something, e.g. 'No text is visible.'"""
    sentences = _re.split(r"(?<=[.!?])\s+", text.strip())
    filtered = [
        s
        for s in sentences
        if not _re.match(r"^(No |There (?:is|are) no |There (?:is|are) nothing )", s, _re.IGNORECASE)
    ]
    return " ".join(filtered)


def load_completed_ids(output_path: Path) -> set[str]:
    """Return the set of sample_ids already written to the output file."""
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                completed.add(obj["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("TASK_ID", "0")))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", os.environ.get("TASK_COUNT", "6")))
    limit = int(os.environ.get("LIMIT", "0")) or None  # 0 = no limit

    print(
        f"Task {task_id}/{task_count - 1}  |  API: {API_BASE_URL}  |  Model: {MODEL_NAME}",
        flush=True,
    )

    # Discover shards and assign this task's subset
    shards = sorted(HQ50K_ROOT.glob("*.tar"))
    if not shards:
        sys.exit(f"No .tar files found in {HQ50K_ROOT}")

    # Collect all (shard_path, member_name) pairs assigned to this task
    all_work: list[tuple[Path, str]] = []
    for shard in shards:
        members = list_shard_members(shard)
        for member_name in members:
            sample_id = f"{shard.stem}/{member_name.replace('.jpg', '')}"
            if stable_task_id(sample_id, task_count) == task_id:
                all_work.append((shard, member_name, sample_id))

    if limit is not None:
        all_work = all_work[:limit]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"candidates_task{task_id:04d}.jsonl"
    failure_path = OUTPUT_DIR / f"failures_task{task_id:04d}.jsonl"

    completed = load_completed_ids(output_path)
    pending = [(s, m, sid) for s, m, sid in all_work if sid not in completed]

    print(
        f"Total: {len(all_work)}  Completed: {len(completed)}  Pending: {len(pending)}",
        flush=True,
    )
    if not pending:
        print("Nothing to do — already complete.", flush=True)
        return

    print("Loading DOCCI index...", flush=True)
    docci = load_docci_index()

    client = OpenAI(base_url=API_BASE_URL, api_key="none")

    n_done = 0
    with (
        output_path.open("a", encoding="utf-8", buffering=1) as out_f,
        failure_path.open("a", encoding="utf-8", buffering=1) as fail_f,
    ):
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start : batch_start + BATCH_SIZE]

            # Load images for the batch
            # Group by shard to open each tar once per batch
            by_shard: dict[str, list[tuple[str, str]]] = {}
            for shard_path, member_name, sample_id in batch:
                by_shard.setdefault(str(shard_path), []).append((member_name, sample_id))

            image_data: dict[str, bytes] = {}
            for shard_str, items in by_shard.items():
                shard_path = Path(shard_str)
                with tarfile.open(shard_path, "r") as tf:
                    for member_name, sample_id in items:
                        fobj = tf.extractfile(member_name)
                        if fobj is not None:
                            image_data[sample_id] = fobj.read()

            # Build one request per sample in the batch
            for shard_path, member_name, sample_id in batch:
                img_bytes = image_data.get(sample_id)
                if img_bytes is None:
                    fail_f.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "error": "Failed to extract image bytes",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                # Deterministic DOCCI selection per sample
                seed = int.from_bytes(blake2b(sample_id.encode(), digest_size=8).digest(), "big")
                rng = random.Random(seed)
                docci_examples = rng.sample(docci, N_DOCCI_EXAMPLES)

                messages = build_messages(img_bytes, docci_examples)

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_TOKENS,
                        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    )
                    caption = strip_absence_sentences((response.choices[0].message.content or "").strip())
                    if not caption:
                        raise ValueError("Empty response from model")
                except Exception as exc:
                    fail_f.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    fail_f.flush()
                    print(f"  FAIL {sample_id}: {exc}", flush=True)
                    continue

                record = {
                    "sample_id": sample_id,
                    "source_sample_id": sample_id,
                    "source_split": "train",
                    "task_type": "caption",
                    "messages": [
                        {"role": "user", "content": "<image_0>"},
                        {"role": "assistant", "content": caption},
                    ],
                    "metadata": {
                        "source_dataset": "hq50k",
                        "data_source": "YangQiee/HQ-50K",
                        "shard": shard_path.name,
                        "member": member_name,
                        "generator_model": MODEL_NAME,
                        "prompt_version": PROMPT_VERSION,
                        "created_at_utc": datetime.now(UTC).isoformat(),
                    },
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1

            total_done = len(completed) + n_done
            print(f"  {total_done}/{len(all_work)} done", flush=True)

    print(f"Finished. Wrote {n_done} new captions to {output_path}", flush=True)


if __name__ == "__main__":
    main()
