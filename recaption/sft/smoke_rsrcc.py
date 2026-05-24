#!/usr/bin/env python3
"""Smoke test for the google_rsrcc loader.

Runs a small generation sample directly on the current node (no SLURM),
then writes a tiny WebDataset shard so the full export path can be verified.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_rsrcc.py

    # With explicit model snapshot:
    SFT_RECAPTION_MODEL_PATH=<snapshot_path> CUDA_VISIBLE_DEVICES=0 \\
        .venv/bin/python scripts/smoke_rsrcc.py

    # Isolated output dir (optional, default is artifacts/candidates/google_rsrcc/):
    SFT_RECAPTION_ARTIFACTS_DIR=/tmp/rsrcc_smoke \\
        CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_rsrcc.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

LIMIT = 4  # samples to generate
BATCH_SIZE = 2  # inference batch size
TASK_COUNT = 1  # single shard for smoke run
TASK_ID = 0
SPLIT = "train"
LOADER_NAME = "google_rsrcc"
WORKER_INDEX = 0

from sft_recaption.config import (
    ModelConfig,
    resolve_model_download_dir,
    resolve_model_reference,
)
from sft_recaption.loaders import create_loader
from sft_recaption.pipeline import generate_candidates
from sft_recaption.runtime import VLLMChatEngine, configure_worker_environment
from sft_recaption.wds_export import export_rsrcc_wds


def _image_prefix(n: int) -> str:
    return "\n".join(f"<image_{i}>" for i in range(n))


def main() -> None:
    configure_worker_environment(WORKER_INDEX)

    model_repo = resolve_model_reference("google/gemma-4-26B-A4B-it")
    print(f"Model : {model_repo}")
    print(f"Loader: {LOADER_NAME}  split={SPLIT}  limit={LIMIT}")
    print()

    loader = create_loader(LOADER_NAME)

    from sft_recaption.config import MANIFESTS_DIR

    manifest_dir = MANIFESTS_DIR / loader.name
    paths = loader.write_manifests(manifest_dir, task_count=TASK_COUNT, split=SPLIT)
    n_records = sum(1 for _ in paths[0].open())
    print(f"Manifest: {paths[0]}  ({n_records} records, using first {LIMIT})")

    engine = VLLMChatEngine(
        ModelConfig(
            model_repo=model_repo,
            tensor_parallel_size=1,
            max_num_seqs=BATCH_SIZE,
            enforce_eager=False,
            download_dir=resolve_model_download_dir(),
        )
    )

    output_path = generate_candidates(
        loader,
        engine,
        task_id=TASK_ID,
        task_count=TASK_COUNT,
        batch_size=BATCH_SIZE,
        split=SPLIT,
        limit=LIMIT,
        model_repo=model_repo,
    )
    print(f"\nCandidates: {output_path}")

    lines = output_path.read_text(encoding="utf-8").splitlines()
    print(f"\n{'=' * 60}")
    print(f"Generated {len(lines)} candidate(s)")
    print(f"{'=' * 60}")

    for i, line in enumerate(lines):
        ex = json.loads(line)
        src = json.loads(ex["metadata"].get("source_fields_json", "{}"))
        question = src.get("question", "")
        answer = src.get("answer", "")
        reasoning = ex["messages"][1]["content"]

        print(f"\n[{i + 1}] {ex['source_sample_id']}")
        print(f"  Q: {question[:180]}")
        print(f"  A (source): {answer}")
        print("  Reasoning (raw, no tags):")
        print(f"    {reasoning[:400]}")
        print("  --- Assembled assistant message ---")
        print(f"    <think>{reasoning[:200]}...</think>")
        print(f"    {answer}")

    # Quick WebDataset export smoke
    print(f"\n{'=' * 60}")
    print("WebDataset export smoke…")
    from sft_recaption.config import CANDIDATES_DIR

    smoke_wds_dir = CANDIDATES_DIR.parent / "wds_smoke" / loader.name
    summary = export_rsrcc_wds(loader, output_dir=smoke_wds_dir, verbose=True)
    print(
        f"\nWDS: {summary['shards_written']} shard(s), {summary['total_written']} sample(s) → {summary['output_dir']}"
    )

    # Verify one entry from the shard
    import tarfile

    shard_files = sorted(smoke_wds_dir.glob("*.tar.gz"))
    if shard_files:
        with tarfile.open(shard_files[0], "r:gz") as tf:
            members = tf.getmembers()
            json_members = [m for m in members if m.name.endswith(".json")]
            if json_members:
                entry = json.loads(tf.extractfile(json_members[0]).read().decode())
                print(f"\nSample from WDS shard ({json_members[0].name}):")
                print(f"  sample_id  : {entry['sample_id']}")
                print(f"  question   : {entry['question'][:120]}")
                print(f"  answer     : {entry['answer']}")
                print(f"  reasoning  : {entry['reasoning'][:200]}…")
                print(f"  assistant  : {entry['messages'][1]['content'][:200]}…")
                print(f"  png files  : {[m.name for m in members if m.name.endswith('.png')][:4]}")


if __name__ == "__main__":
    main()
