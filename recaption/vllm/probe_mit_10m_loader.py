#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import islice
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent.parent
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

from main import (
    DEFAULT_CHAT_TEMPLATE_KWARGS,
    DEFAULT_DTYPE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_LIMIT_MM_PER_PROMPT,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MM_PROCESSOR_KWARGS,
    DEFAULT_MODEL_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_MODEL_REPO,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    DEFAULT_TRUST_REMOTE_CODE,
    get_loader_attr,
    load_loader,
    resolve_messages,
    resolve_model_path,
)
from vllm import LLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=64)
    parser.add_argument("--count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["RECAPTION_TARGET_LANG"] = args.lang

    loader = load_loader(args.loader)
    samples = list(islice(loader.iter_samples(args.task_id, args.task_count), args.count))
    if not samples:
        raise RuntimeError("No samples available for probe")

    model_dir = str(get_loader_attr(loader, "model_dir", DEFAULT_MODEL_DIR))
    model_repo = str(get_loader_attr(loader, "model_repo", DEFAULT_MODEL_REPO))
    model_cache_dir = get_loader_attr(loader, "model_cache_dir", DEFAULT_MODEL_CACHE_DIR)
    tensor_parallel_size = int(get_loader_attr(loader, "tensor_parallel_size", DEFAULT_TENSOR_PARALLEL_SIZE))
    gpu_memory_utilization = float(get_loader_attr(loader, "gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION))
    max_model_len = int(get_loader_attr(loader, "max_model_len", DEFAULT_MAX_MODEL_LEN))
    max_num_batched_tokens = int(get_loader_attr(loader, "max_num_batched_tokens", DEFAULT_MAX_NUM_BATCHED_TOKENS))
    mm_processor_kwargs = get_loader_attr(loader, "mm_processor_kwargs", DEFAULT_MM_PROCESSOR_KWARGS)
    limit_mm_per_prompt = get_loader_attr(loader, "limit_mm_per_prompt", DEFAULT_LIMIT_MM_PER_PROMPT)
    chat_template_kwargs = get_loader_attr(loader, "chat_template_kwargs", DEFAULT_CHAT_TEMPLATE_KWARGS)
    dtype = str(get_loader_attr(loader, "dtype", DEFAULT_DTYPE))
    trust_remote_code = bool(get_loader_attr(loader, "trust_remote_code", DEFAULT_TRUST_REMOTE_CODE))
    sampling = get_loader_attr(loader, "sampling_params", None)
    llm_kwargs = dict(get_loader_attr(loader, "llm_kwargs", {}))
    model_path = resolve_model_path(model_repo, model_cache_dir)

    conversations = [resolve_messages(loader, sample) for sample in samples]

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        download_dir=model_dir,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        trust_remote_code=trust_remote_code,
        max_num_seqs=len(samples),
        **({"limit_mm_per_prompt": limit_mm_per_prompt} if limit_mm_per_prompt is not None else {}),
        **({"mm_processor_kwargs": mm_processor_kwargs} if mm_processor_kwargs is not None else {}),
        **llm_kwargs,
    )
    outputs = llm.chat(
        conversations,
        sampling_params=sampling,
        use_tqdm=False,
        chat_template_kwargs=chat_template_kwargs,
    )

    for sample, output in zip(samples, outputs):
        payload = {
            "sample_id": sample["sample_id"],
            "conditioning_text": sample["conditioning_text"],
            "metadata": sample.get("metadata") or {},
            "caption": output.outputs[0].text.strip(),
        }
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
