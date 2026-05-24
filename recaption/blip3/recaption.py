#!/usr/bin/env python3
"""Recaption worker: reads tar files, generates grounded captions, writes output JSONL."""

import argparse
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams

MODEL_PATH = os.environ.get("RECAPTION_MODEL_PATH", "")
WORK_DIR = Path(os.environ.get("RECAPTION_WORK_DIR", "/tmp/recaption_blip3"))
OUTPUT_DIR = WORK_DIR / "output"
BATCH_SIZE = 32
MAX_LLM_EDGE = 512
MIN_IMAGE_EDGE = 32

SYSTEM_PROMPT = (
    "You are a visual annotator. Output a single paragraph of natural sentences describing the image. "
    "For every object, region, surface, or element—foreground and background—annotate it inline as: "
    "<object>label</object><bbox>[x_min, y_min, x_max, y_max]</bbox> "
    "where coordinates are integers in [0, 1000] (top-left 0,0; bottom-right 1000,1000). "
    'Example: "The scene shows a <object>red sofa</object><bbox>[120, 400, 650, 850]</bbox> '
    'against a <object>white wall</object><bbox>[0, 0, 1000, 600]</bbox>." '
    "Cover as many distinct items as possible. Be specific about colors, shapes, and positions. "
    "Output only the annotated paragraph—no reasoning, no lists, no headers."
)

_BBOX_RE = re.compile(r"<bbox>\[([^\]]+)\]</bbox>")


def fix_caption(caption: str) -> str:
    def _fix(m):
        try:
            coords = [float(x.strip()) for x in m.group(1).split(",")]
            if len(coords) != 4:
                return m.group(0)
            # If all coords <= 1.0, model used normalized [0,1] scale — convert
            if all(c <= 1.0 for c in coords):
                coords = [c * 1000 for c in coords]
            coords = [max(0, min(1000, round(c))) for c in coords]
            return f"<bbox>[{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]</bbox>"
        except Exception:
            return m.group(0)

    return _BBOX_RE.sub(_fix, caption)


def resize_for_llm(img: Image.Image) -> Image.Image:
    w, h = img.size
    if max(w, h) <= MAX_LLM_EDGE:
        return img
    scale = MAX_LLM_EDGE / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def build_conversation(llm_image: Image.Image, orig_caption: str) -> list[dict]:
    user_text = (
        "Annotate this image with inline bounding boxes for every visible element. Write a single paragraph only."
    )
    if orig_caption:
        user_text += f"\n\nContext (reference only, do not reproduce): {orig_caption[:300]}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": llm_image},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def flush_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    keys: list[str],
    source_tars: list[str],
    llm_images: list[Image.Image],
    orig_captions: list[str],
    out_f,
) -> int:
    conversations = [build_conversation(img, cap) for img, cap in zip(llm_images, orig_captions)]
    try:
        outputs = llm.chat(
            conversations,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )
    except Exception as e:
        print(f"  batch inference error: {e}", file=sys.stderr)
        return 0

    written = 0
    for key, source_tar, output in zip(keys, source_tars, outputs):
        caption = fix_caption(output.outputs[0].text.strip())
        record = {"key": key, "source_tar": source_tar, "caption": caption}
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
    out_f.flush()
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-map", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--tar-start", type=int, default=0)
    parser.add_argument("--tar-end", type=int, default=None)
    args = parser.parse_args()

    with open(args.chunk_map) as f:
        chunk_map = json.load(f)

    tar_paths = chunk_map.get(str(args.chunk_id), [])
    tar_paths = tar_paths[args.tar_start : args.tar_end]
    if not tar_paths:
        print(f"Chunk {args.chunk_id}: no tars assigned, exiting.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUTPUT_DIR / f"{args.chunk_id:04d}.jsonl"

    # Resume: build set of already-done (key, source_tar) pairs
    done_keys: set[tuple[str, str]] = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done_keys.add((r["key"], r["source_tar"]))
                    except Exception:
                        pass
        if done_keys:
            print(
                f"Chunk {args.chunk_id}: resuming, {len(done_keys)} samples already done.",
                flush=True,
            )

    print(f"Chunk {args.chunk_id}: loading model...", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        max_num_seqs=256,
    )
    sampling_params = SamplingParams(temperature=0.2, max_tokens=400)
    print(
        f"Chunk {args.chunk_id}: model ready, processing {len(tar_paths)} tars...",
        flush=True,
    )

    total = 0
    with open(jsonl_path, "a", buffering=1) as out_f:  # line-buffered: each write() is one atomic O_APPEND syscall
        keys, source_tars, llm_images, orig_captions = [], [], [], []

        for tar_path in tar_paths:
            print(f"  {tar_path}", flush=True)
            try:
                with tarfile.open(tar_path, "r") as tf:
                    name_to_member = {m.name: m for m in tf.getmembers()}
                    jpg_members = sorted(
                        (m for m in name_to_member.values() if m.name.endswith(".jpg")),
                        key=lambda m: m.name,
                    )

                    for jpg_member in jpg_members:
                        if args.max_samples is not None and total + len(keys) >= args.max_samples:
                            break

                        key = jpg_member.name[:-4]
                        if (key, tar_path) in done_keys:
                            continue

                        try:
                            raw_jpg = tf.extractfile(jpg_member).read()
                            img = Image.open(io.BytesIO(raw_jpg)).convert("RGB")
                            if min(img.size) < MIN_IMAGE_EDGE:
                                continue
                        except Exception:
                            continue

                        orig_caption = ""
                        txt_member = name_to_member.get(key + ".txt")
                        if txt_member:
                            try:
                                orig_caption = (
                                    tf.extractfile(txt_member).read().decode("utf-8", errors="ignore").strip()
                                )
                            except Exception:
                                pass

                        keys.append(key)
                        source_tars.append(tar_path)
                        llm_images.append(resize_for_llm(img))
                        orig_captions.append(orig_caption)

                        if len(keys) >= BATCH_SIZE:
                            total += flush_batch(
                                llm,
                                sampling_params,
                                keys,
                                source_tars,
                                llm_images,
                                orig_captions,
                                out_f,
                            )
                            keys, source_tars, llm_images, orig_captions = (
                                [],
                                [],
                                [],
                                [],
                            )
                            print(f"    {total} samples written", flush=True)
            except Exception as e:
                print(f"  error opening {tar_path}: {e}", file=sys.stderr)
                continue

        if keys:
            total += flush_batch(
                llm,
                sampling_params,
                keys,
                source_tars,
                llm_images,
                orig_captions,
                out_f,
            )

    print(f"Chunk {args.chunk_id}: done, {total} new samples → {jsonl_path}", flush=True)


if __name__ == "__main__":
    main()
