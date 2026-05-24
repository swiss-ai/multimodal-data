#!/usr/bin/env python3
"""Rewrite long gpt turns using Qwen3.6-27B: let Qwen decide whether to add <think> tags."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from rouge_l import rouge_l_f1
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_PATH = "/tmp/models/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
TOKENIZER_PATH = MODEL_PATH
BATCH_SIZE = 32
MAX_INPUT_CHARS = 6000  # truncate answer in prompt to leave room for output

SYSTEM_PROMPT = (
    "You are a structural reformatter. These answers originally had a <reasoning>...</reasoning> "
    "block that was stripped. Restore it: wrap the reasoning/exploration/working-out in "
    "<reasoning>...</reasoning> and leave only the final answer after. No newlines around the tags — "
    "format is exactly: <reasoning>reasoning</reasoning>final answer. Copy every word exactly. "
    "If there is no separable reasoning (answer is already direct), output it EXACTLY unchanged."
)

# Few-shot examples. Uses <reasoning> tags to avoid conflict with Qwen's special <think> tokens.
# write_result() replaces <reasoning>/<\/reasoning> → <think>/<\/think> in the saved output.
FEW_SHOT = [
    {
        "role": "user",
        "content": (
            "Question: If a train travels 120 km in 2 hours, what is its average speed?\n\nAnswer:\n"
            "Hmm, let me work this out. Speed = distance / time. Distance is 120 km, time is 2 hours. "
            "So speed = 120 / 2 = 60 km/h. The average speed is 60 km/h."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<reasoning>Hmm, let me work this out. Speed = distance / time. Distance is 120 km, "
            "time is 2 hours. So speed = 120 / 2 = 60 km/h.</reasoning>The average speed is 60 km/h."
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: What is photosynthesis?\n\nAnswer:\n"
            "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide "
            "into glucose and oxygen."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide "
            "into glucose and oxygen."
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: Solve for x: 3x + 7 = 22\n\nAnswer:\n"
            "Let me solve this step by step. Subtract 7 from both sides: 3x = 22 - 7 = 15. "
            "Then divide both sides by 3: x = 15 / 3 = 5.\n\n**Answer**: x = 5"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<reasoning>Let me solve this step by step. Subtract 7 from both sides: 3x = 22 - 7 = 15. "
            "Then divide both sides by 3: x = 15 / 3 = 5.</reasoning>**Answer**: x = 5"
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: Did Team A score more goals than Team B? Answer with Yes or No.\n\nAnswer:\n"
            "用户需要比较Team A和Team B的进球数。Team A得了3球，Team B得了2球，所以Team A更多，答案是Yes。"
            "To determine which team scored more:\n\n"
            "1. Team A scored 3 goals.\n"
            "2. Team B scored 2 goals.\n"
            "3. Since 3 > 2, Team A scored more.\n\n"
            "Answer: Yes"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "<reasoning>用户需要比较Team A和Team B的进球数。Team A得了3球，Team B得了2球，所以Team A更多，答案是Yes。"
            "To determine which team scored more:\n\n"
            "1. Team A scored 3 goals.\n"
            "2. Team B scored 2 goals.\n"
            "3. Since 3 > 2, Team A scored more.</reasoning>Answer: Yes"
        ),
    },
]


def build_conversation(question: str, answer: str) -> list[dict]:
    truncated = answer if len(answer) <= MAX_INPUT_CHARS else answer[:MAX_INPUT_CHARS] + "\n...[truncated]"
    return (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEW_SHOT
        + [{"role": "user", "content": f"Question: {question}\n\nAnswer:\n{truncated}"}]
    )


def write_result(rec: dict, rewritten: str, tokenizer, out_f) -> None:
    # Swap proxy tags → final <think> tags (Qwen suppresses <think> during sampling)
    final = rewritten.replace("<reasoning>", "<think>").replace("</reasoning>", "</think>")
    ref_tokens = np.array(tokenizer.encode(rec["answer"])[:2048], dtype=np.int32)
    pred_tokens = np.array(tokenizer.encode(final)[:2048], dtype=np.int32)
    score = float(rouge_l_f1(ref_tokens, pred_tokens))
    result = {
        "id": rec["id"],
        "file": rec["file"],
        "row_idx": rec["row_idx"],
        "turn_idx": rec["turn_idx"],
        "rewritten": final,
        "rouge_l": round(score, 4),
    }
    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")


def flush_batch(llm, params, tokenizer, batch_records, out_f) -> int:
    conversations = [build_conversation(r["question"], r["answer"]) for r in batch_records]
    try:
        outputs = llm.chat(
            conversations,
            params,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as e:
        print(f"  batch error: {e}", file=sys.stderr)
        return 0

    for rec, output in zip(batch_records, outputs):
        write_result(rec, output.outputs[0].text.strip(), tokenizer, out_f)
    out_f.flush()
    return len(batch_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", required=True, help="Input chunk JSONL")
    parser.add_argument("--output", required=True, help="Output rewrites JSONL")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    chunk_path = Path(args.chunk)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line)["id"])
                    except Exception:
                        pass
        if done_ids:
            print(f"Resuming: {len(done_ids)} samples already done", flush=True)

    records = []
    with open(chunk_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec["id"] not in done_ids:
                    records.append(rec)

    print(f"Chunk {chunk_path.name}: {len(records)} samples to process", flush=True)
    if not records:
        print("Nothing to do, exiting.", flush=True)
        return

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    # Warm up numba JIT
    dummy = np.array([1, 2, 3], dtype=np.int32)
    rouge_l_f1(dummy, dummy)

    print("Loading model...", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=8192,
        max_num_seqs=64,
    )
    params = SamplingParams(temperature=0.3, max_tokens=4096)
    print("Model ready.", flush=True)

    total = 0
    with open(output_path, "a", buffering=1) as out_f:
        batch = []
        for rec in records:
            batch.append(rec)
            if len(batch) >= args.batch_size:
                total += flush_batch(llm, params, tokenizer, batch, out_f)
                batch = []
                print(f"  {total} done", flush=True)
        if batch:
            total += flush_batch(llm, params, tokenizer, batch, out_f)

    print(f"Chunk {chunk_path.name}: done, {total} rewrites → {output_path}", flush=True)


if __name__ == "__main__":
    main()
