#!/usr/bin/env python3
"""
Batch caption generation using vllm.
Parallelizable via SLURM job array, one task per tar file (or range of tars).
"""

import base64
import json
import os
from io import BytesIO

from loaders.wds import WdsLoader
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

INPUT_DIR = os.environ.get("WDS_INPUT_DIR", "/tmp/shared/EgoPAT3Dv2/output_wds")
OUTPUT_DIR = os.environ.get("WDS_OUTPUT_DIR", "/tmp/toolbox/caption_gen/logs/egopat")
PROMPT_PATH = os.environ.get("PROMPT_PATH", "/tmp/toolbox/caption_gen/prompts/ego_3.txt")

IMAGE_KEYS = ("img0.jpg", "img1.jpg", "img2.jpg")

MODEL = "moonshotai/Kimi-VL-A3B-Thinking-2506"
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models")

SAMPLING = SamplingParams(
    temperature=0.8,
    max_tokens=32768,
)

TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 131072
MAX_NUM_SEQS = 8
BATCH_SIZE = 128
GPU_MEMORY_UTIL = 0.95

BOT = "◁think▷"
EOT = "◁/think▷"


def extract_thinking_and_summary(text: str) -> tuple[str, str]:
    if BOT in text and EOT not in text:
        return "", ""
    if EOT in text:
        thinking = text[text.index(BOT) + len(BOT) : text.index(EOT)].strip()
        summary = text[text.index(EOT) + len(EOT) :].strip()
        return thinking, summary
    return "", text


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_str)))


def build_request(pil_images: list[Image.Image], prompt: str, processor) -> dict:
    content = [{"type": "image", "image": ""} for _ in pil_images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return {
        "prompt": text,
        "multi_modal_data": {"image": pil_images},
    }


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
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        trust_remote_code=True,
        max_num_seqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        limit_mm_per_prompt={"image": 256},
    )
    processor = AutoProcessor.from_pretrained(MODEL, cache_dir=MODEL_DIR, trust_remote_code=True)

    print("Generating captions...")
    captions: dict[str, str] = {}
    for bi, batch in enumerate(loader.stream(BATCH_SIZE)):
        keys = [k for k, _ in batch]
        requests = [build_request([b64_to_pil(b) for b in b64s], prompt, processor) for _, b64s in batch]
        outputs = llm.generate(requests, SAMPLING, use_tqdm=False)
        for k, out in zip(keys, outputs):
            _, summary = extract_thinking_and_summary(out.outputs[0].text)
            captions[k] = summary
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
