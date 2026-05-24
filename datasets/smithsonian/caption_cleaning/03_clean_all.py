#!/usr/bin/env python3
"""
03_clean_all.py

Distributed Smithsonian caption cleaner. Run as a SLURM array job.

For each assigned tar shard (round-robin by global tar index % num_workers):
  1. Read all samples (.jpg + .txt + .json triplets).
  2. Pre-filter: skip samples with len(txt) < MIN_LEN.
  3. Run LLM (Gemma 4 31B-it, text-only) on passing captions.
  4. Post-filter: skip samples with len(cleaned) < MIN_LEN.
  5. Write a new output tar at OUT_ROOT/{subset}/{tarname} containing only
     the kept samples: original .jpg + .json, cleaned .txt.
     Samples that fail pre- or post-filter are omitted entirely.

Usage:
    python 03_clean_all.py --worker-id N --num-workers M
"""

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from prompts import PROMPT_BY_SUBSET  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WDS_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian")
OUT_ROOT = Path("/path/to/data/vision-datasets/processed/smithsonian_cleaned2")
CHECKPOINT_DIR = Path(__file__).parent.parent / "data" / "clean_all_checkpoints2"

MODEL_PATH = "/tmp/models/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445"
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.80
MAX_TOKENS = 1024
TEMPERATURE = 0.1
TOP_P = 0.9
MIN_LEN = 200

APPROVED_SUBSETS = [
    "tier1/nmaahc",
    "tier1/npg",
    "tier2/design/chndm",
    "tier2/history/nasm",
    "tier2/history/nmah",
    "tier2/other/acm",
    "tier2/other/npm",
    "tier2/other/sia",
]


# ---------------------------------------------------------------------------
# Tar I/O
# ---------------------------------------------------------------------------


def discover_tars() -> list[tuple[str, Path]]:
    """Return sorted list of (subset_key, tar_path) for all approved subsets."""
    result = []
    for subset in APPROVED_SUBSETS:
        d = WDS_ROOT / subset
        for tar in sorted(d.glob("*.tar")):
            result.append((subset, tar))
    return result


def read_tar_samples(tar_path: Path) -> list[dict]:
    """
    Stream through a tar once, collecting all (jpg, txt, json) triplets.
    Returns list of dicts: {key, jpg, txt, json_bytes}.
    """
    buffers: dict[str, dict] = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            if "." not in name:
                continue
            stem, ext = name.rsplit(".", 1)
            f = tf.extractfile(member)
            if f is None:
                continue
            raw = f.read()
            if stem not in buffers:
                buffers[stem] = {}
            if ext == "jpg":
                buffers[stem]["jpg"] = raw
            elif ext == "txt":
                buffers[stem]["txt"] = raw.decode("utf-8", errors="replace").strip()
            elif ext == "json":
                buffers[stem]["json_bytes"] = raw

    samples = []
    for key, parts in buffers.items():
        if "jpg" in parts and "txt" in parts and "json_bytes" in parts:
            samples.append({"key": key, **parts})
    return samples


def write_output_tar(out_path: Path, kept: list[dict]) -> None:
    """Write kept samples to a new tar. Preserves member order: jpg, txt, json."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w") as tf:
        for s in kept:
            for ext, content in [
                ("jpg", s["jpg"]),
                ("txt", s["cleaned_txt"].encode("utf-8")),
                ("json", s["json_bytes"]),
            ]:
                info = tarfile.TarInfo(name=f"{s['key']}.{ext}")
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> set[str]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"Checkpoint loaded: {len(data)} tars already done")
        return set(data)
    return set()


def save_checkpoint(path: Path, completed: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(completed), indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def build_prompt(subset_key: str, caption: str) -> list[dict]:
    template = PROMPT_BY_SUBSET.get(subset_key, PROMPT_BY_SUBSET["default"])
    return [{"role": "user", "content": template.format(caption=caption)}]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    args = parser.parse_args()

    worker_id = args.worker_id
    num_workers = args.num_workers

    checkpoint_path = CHECKPOINT_DIR / f"worker_{worker_id:03d}.json"
    completed = load_checkpoint(checkpoint_path)

    all_tars = discover_tars()
    my_tars = [(s, p) for i, (s, p) in enumerate(all_tars) if i % num_workers == worker_id]
    pending = [(s, p) for s, p in my_tars if str(p) not in completed]

    print(
        f"[worker {worker_id}/{num_workers}] assigned {len(my_tars)} tars, {len(pending)} pending",
        flush=True,
    )
    if not pending:
        print("Nothing to do.")
        return

    print(f"[worker {worker_id}] Loading model ...", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS)

    total_read = total_pre_filtered = total_llm = total_post_filtered = total_kept = 0

    for tar_idx, (subset_key, tar_path) in enumerate(pending, 1):
        print(
            f"[worker {worker_id}] [{tar_idx}/{len(pending)}] {subset_key}/{tar_path.name}",
            flush=True,
        )
        out_path = OUT_ROOT / subset_key / tar_path.name

        # --- Read ---
        samples = read_tar_samples(tar_path)
        total_read += len(samples)

        # --- Pre-filter ---
        to_clean = [s for s in samples if len(s["txt"]) >= MIN_LEN]
        n_pre = len(samples) - len(to_clean)
        total_pre_filtered += n_pre
        total_llm += len(to_clean)
        print(
            f"  {len(samples)} samples: {len(to_clean)} to LLM, {n_pre} pre-filtered",
            flush=True,
        )

        # --- LLM ---
        kept = []
        if to_clean:
            conversations = [build_prompt(subset_key, s["txt"]) for s in to_clean]
            outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=True)
            n_post = 0
            for s, out in zip(to_clean, outputs):
                cleaned = out.outputs[0].text.strip()
                if len(cleaned) < MIN_LEN:
                    n_post += 1
                else:
                    kept.append(
                        {
                            "key": s["key"],
                            "jpg": s["jpg"],
                            "cleaned_txt": cleaned,
                            "json_bytes": s["json_bytes"],
                        }
                    )
            total_post_filtered += n_post
            print(f"  kept {len(kept)}, post-filtered {n_post}", flush=True)

        total_kept += len(kept)

        # --- Write ---
        if kept:
            write_output_tar(out_path, kept)
            print(f"  -> {out_path} ({len(kept)} samples)", flush=True)
        else:
            print("  -> no samples kept, tar skipped", flush=True)

        completed.add(str(tar_path))
        save_checkpoint(checkpoint_path, completed)

    print(f"\n[worker {worker_id}] === Done ===")
    print(f"  Read:          {total_read}")
    print(f"  Pre-filtered:  {total_pre_filtered}  (txt < {MIN_LEN}c)")
    print(f"  LLM calls:     {total_llm}")
    print(f"  Post-filtered: {total_post_filtered}  (cleaned < {MIN_LEN}c)")
    print(f"  Kept:          {total_kept}")


if __name__ == "__main__":
    main()
