#!/usr/bin/env python3
"""
Batch caption generation using vllm.
Parallelizable via SLURM job array, one task per tar file (or range of tars).
"""

import json
import os

from loaders.wds import WdsLoader
from vllm import LLM, SamplingParams

INPUT_DIR = os.environ.get("WDS_INPUT_DIR", "/tmp/shared/EgoPAT3Dv2/output_wds")
OUTPUT_DIR = os.environ.get("WDS_OUTPUT_DIR", "/tmp/toolbox/caption_gen/logs/egopat")
PROMPT_PATH = os.environ.get("PROMPT_PATH", "/tmp/toolbox/caption_gen/prompts/ego_3.txt")

IMAGE_KEYS = ("img0.jpg", "img1.jpg", "img2.jpg")

MODEL = "moonshotai/Kimi-VL-A3B-Instruct"
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models")

SAMPLING = SamplingParams(
    temperature=0.1,
    top_p=0.95,
    max_tokens=2048,
)

TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTIL = 0.95
MAX_MODEL_LEN = 16384
BATCH_SIZE = 128


def build_messages(b64_images: list[str], prompt: str) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in b64_images]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    out_path = os.path.join(OUTPUT_DIR, f"captions_task{task_id:04d}.json")
    if os.path.exists(out_path):
        print(f"Output already exists, skipping: {out_path}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    loader = WdsLoader(task_id, task_count, INPUT_DIR, IMAGE_KEYS)
    with open(PROMPT_PATH) as f:
        prompt = f.read().strip()

    print(f"TASK {task_id}/{task_count - 1}")
    print("Loading model...")
    llm = LLM(
        model=MODEL,
        download_dir=MODEL_DIR,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        trust_remote_code=True,
    )

    print("Generating captions...")
    captions: dict[str, str] = {}
    for bi, batch in enumerate(loader.stream(BATCH_SIZE)):
        keys = [k for k, _ in batch]
        conversations = [build_messages(imgs, prompt) for _, imgs in batch]
        outputs = llm.chat(conversations, SAMPLING, use_tqdm=False)
        for k, out in zip(keys, outputs):
            captions[k] = out.outputs[0].text
        print(f"{len(captions)} samples done...")
        if bi % 10 == 0:
            with open(out_path, "w") as f:
                f.write(json.dumps(captions, indent=2, ensure_ascii=False))

    assert all(v.strip() for v in captions.values()), "Some captions are empty"
    with open(out_path, "w") as f:
        f.write(json.dumps(captions, indent=2, ensure_ascii=False))
    print(f"{len(captions)} samples total — saved to {out_path}")


if __name__ == "__main__":
    main()
