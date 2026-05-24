#!/usr/bin/env python3

import base64
import json
import os
import re
import time
from importlib import import_module
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Thread

from vllm import LLM, SamplingParams

DEFAULT_MODEL_DIR = "/tmp/models"
DEFAULT_MODEL_REPO = "Qwen/Qwen3.5-9B"
DEFAULT_MODEL_CACHE_DIR = Path(DEFAULT_MODEL_DIR) / "models--Qwen--Qwen3.5-9B"
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.94
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_BATCH_SIZE = 96
DEFAULT_PREFETCH_BATCHES = 1
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_MM_PROCESSOR_KWARGS = {
    "min_pixels": 28 * 28,
    "max_pixels": 1280 * 28 * 28,
}
DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
DEFAULT_LIMIT_MM_PER_PROMPT = {"image": 1}
DEFAULT_DTYPE = "bfloat16"
DEFAULT_TRUST_REMOTE_CODE = True

DEFAULT_SAMPLING = SamplingParams(
    temperature=0.3,
    top_p=0.7,
    top_k=20,
    max_tokens=256,
)


def load_processed_sample_ids(path: Path) -> set[str]:
    processed_sample_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            sample_id = payload["sample_id"]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"bad sample_id {path}:{line_number}: {sample_id!r}")
            processed_sample_ids.add(sample_id)
    return processed_sample_ids


def load_loader(loader_module_name: str):
    loader_module = import_module(loader_module_name)
    return loader_module.loader


def resolve_user_prompt(loader, raw_sample: dict) -> str:
    prompt_source = getattr(loader, "build_prompt", None)
    if prompt_source is None:
        prompt_source = getattr(loader, "prompt", None)

    prompt = prompt_source(raw_sample) if callable(prompt_source) else prompt_source
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Loader {getattr(loader, 'name', loader)!r} returned an invalid prompt")
    return prompt


def build_multimodal_messages(
    image_data_url: str,
    user_prompt: str,
    system_prompt: str | None = None,
) -> list[dict]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": user_prompt},
            ],
        }
    )
    return messages


def to_data_url(image_bytes: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def validate_messages(messages: object) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Loader returned invalid chat messages: {messages!r}")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError(f"Loader returned non-dict chat message: {message!r}")
    return messages


def resolve_messages(loader, raw_sample: dict) -> list[dict]:
    build_messages = getattr(loader, "build_messages", None)
    if callable(build_messages):
        return validate_messages(build_messages(raw_sample))

    return build_multimodal_messages(
        to_data_url(
            bytes(raw_sample["image_bytes"]),
            raw_sample.get("media_type", "image/jpeg"),
        ),
        resolve_user_prompt(loader, raw_sample),
        getattr(loader, "system_prompt", None),
    )


def resolve_model_path(model_repo: str, model_cache_dir: Path) -> str:
    ref_path = model_cache_dir / "refs" / "main"
    if not ref_path.exists():
        return model_repo

    snapshot_name = ref_path.read_text(encoding="utf-8").strip()
    snapshot_path = model_cache_dir / "snapshots" / snapshot_name
    if snapshot_path.exists():
        return str(snapshot_path)

    return model_repo


def get_loader_attr(loader, attr_name: str, default):
    return getattr(loader, attr_name, default)


THINKING_PATTERN = re.compile(
    r"<\|channel\|>thought\s*(.*?)\s*<channel\|>",
    flags=re.DOTALL,
)


def strip_thinking(text: str) -> tuple[str, str]:
    thinking_chunks = THINKING_PATTERN.findall(text)
    without_thinking = THINKING_PATTERN.sub("", text)
    cleaned = without_thinking.strip()
    return cleaned, "\n\n".join(chunk.strip() for chunk in thinking_chunks if chunk.strip())


def resolve_generation_seconds(output) -> float | None:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    arrival_time = getattr(metrics, "arrival_time", 0.0) or 0.0
    last_token_ts = getattr(metrics, "last_token_ts", 0.0) or 0.0
    if arrival_time and last_token_ts and last_token_ts >= arrival_time:
        return last_token_ts - arrival_time
    return None


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def build_transformers_messages(messages: list[dict]) -> list[dict]:
    transformed = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            transformed.append({"role": message["role"], "content": content})
            continue

        if not isinstance(content, list):
            raise ValueError(f"Unsupported transformers message content: {content!r}")

        transformed_content = []
        for item in content:
            item_type = item.get("type")
            if item_type == "image_url":
                transformed_content.append({"type": "image"})
            elif item_type == "text":
                transformed_content.append({"type": "text", "text": item["text"]})
            else:
                raise ValueError(f"Unsupported transformers content item: {item!r}")

        transformed.append({"role": message["role"], "content": transformed_content})
    return transformed


def load_transformers_runtime(
    model_path: str,
    dtype: str,
    trust_remote_code: bool,
    extra_model_kwargs: dict,
):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        **extra_model_kwargs,
    )
    return processor, model


def run_transformers_batch(
    samples: list[dict],
    conversations: list[list[dict]],
    processor,
    model,
    sampling,
    chat_template_kwargs: dict,
) -> list[dict]:
    import torch
    from PIL import Image

    outputs = []
    model_device = next(model.parameters()).device

    for sample, conversation in zip(samples, conversations):
        image = Image.open(BytesIO(sample["image_bytes"]))
        rendered_prompt = processor.apply_chat_template(
            build_transformers_messages(conversation),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        inputs = processor(
            text=rendered_prompt,
            images=[image],
            return_tensors="pt",
        )
        inputs = {key: value.to(model_device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        generate_kwargs = {
            "max_new_tokens": sampling.max_tokens,
            "do_sample": bool(sampling.temperature and sampling.temperature > 0),
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = sampling.temperature
            generate_kwargs["top_p"] = sampling.top_p
            generate_kwargs["top_k"] = sampling.top_k

        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, **generate_kwargs)
        generation_seconds = time.perf_counter() - started

        generated_ids = generated[0][input_len:]
        outputs.append(
            {
                "text": processor.decode(
                    generated_ids,
                    skip_special_tokens=False,
                ).strip(),
                "generation_seconds": generation_seconds,
                "prompt_tokens": int(input_len),
                "completion_tokens": int(generated_ids.shape[-1]),
            }
        )

    return outputs


def main() -> None:
    loader_module_name = os.environ["RECAPTION_LOADER"]
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    task_count = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    loader = load_loader(loader_module_name)
    model_dir = str(get_loader_attr(loader, "model_dir", DEFAULT_MODEL_DIR))
    model_repo = str(get_loader_attr(loader, "model_repo", DEFAULT_MODEL_REPO))
    model_cache_dir = Path(get_loader_attr(loader, "model_cache_dir", DEFAULT_MODEL_CACHE_DIR))
    tensor_parallel_size = int(get_loader_attr(loader, "tensor_parallel_size", DEFAULT_TENSOR_PARALLEL_SIZE))
    gpu_memory_utilization = float(get_loader_attr(loader, "gpu_memory_utilization", DEFAULT_GPU_MEMORY_UTILIZATION))
    max_model_len = int(get_loader_attr(loader, "max_model_len", DEFAULT_MAX_MODEL_LEN))
    batch_size = int(get_loader_attr(loader, "batch_size", DEFAULT_BATCH_SIZE))
    prefetch_batches = int(get_loader_attr(loader, "prefetch_batches", DEFAULT_PREFETCH_BATCHES))
    max_num_batched_tokens = int(get_loader_attr(loader, "max_num_batched_tokens", DEFAULT_MAX_NUM_BATCHED_TOKENS))
    mm_processor_kwargs = get_loader_attr(loader, "mm_processor_kwargs", DEFAULT_MM_PROCESSOR_KWARGS)
    limit_mm_per_prompt = get_loader_attr(loader, "limit_mm_per_prompt", DEFAULT_LIMIT_MM_PER_PROMPT)
    chat_template_kwargs = get_loader_attr(loader, "chat_template_kwargs", DEFAULT_CHAT_TEMPLATE_KWARGS)
    dtype = str(get_loader_attr(loader, "dtype", DEFAULT_DTYPE))
    trust_remote_code = bool(get_loader_attr(loader, "trust_remote_code", DEFAULT_TRUST_REMOTE_CODE))
    sampling = get_loader_attr(loader, "sampling_params", DEFAULT_SAMPLING)
    extra_llm_kwargs = dict(get_loader_attr(loader, "llm_kwargs", {}))
    inference_backend = str(get_loader_attr(loader, "inference_backend", "vllm"))
    transformers_model_kwargs = dict(get_loader_attr(loader, "transformers_model_kwargs", {}))
    model_path = resolve_model_path(model_repo, model_cache_dir)

    loader.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = loader.output_dir / f"captions_task{task_id:04d}.jsonl"
    tmp_path = loader.output_dir / f"captions_task{task_id:04d}.jsonl.tmp"

    if final_path.exists():
        print(f"Final output already exists, skipping task: {final_path}")
        return

    if tmp_path.exists():
        processed_sample_ids = load_processed_sample_ids(tmp_path)
        print(f"Resuming {tmp_path} with {len(processed_sample_ids)} completed")
    else:
        tmp_path.touch()
        processed_sample_ids = set()

    print(
        "Runtime config: "
        f"loader={loader_module_name} "
        f"dataset={getattr(loader, 'name', loader_module_name)} "
        f"backend={inference_backend} "
        f"task_id={task_id} "
        f"task_count={task_count} "
        f"output_dir={loader.output_dir} "
        f"batch_size={batch_size} "
        f"prefetch_batches={prefetch_batches} "
        f"tensor_parallel_size={tensor_parallel_size} "
        f"max_model_len={max_model_len} "
        f"max_num_batched_tokens={max_num_batched_tokens} "
        f"gpu_memory_utilization={gpu_memory_utilization:.3f} "
        f"dtype={dtype} "
        f"trust_remote_code={trust_remote_code} "
        f"chat_template_kwargs={chat_template_kwargs} "
        f"limit_mm_per_prompt={limit_mm_per_prompt} "
        f"mm_processor_kwargs={mm_processor_kwargs} "
        f"max_tokens={sampling.max_tokens} "
        f"model_path={model_path}"
    )

    iter_samples_resume = getattr(loader, "iter_samples_resume", None)
    if callable(iter_samples_resume):
        sample_iter = iter_samples_resume(
            task_id,
            task_count,
            processed_count=len(processed_sample_ids),
            processed_sample_ids=processed_sample_ids,
        )
    else:
        sample_iter = loader.iter_samples(task_id, task_count)
    timings = {"load": 0.0, "encode": 0.0, "model": 0.0}
    total_started = time.perf_counter()

    def prepare_batch():
        samples = []
        conversations = []

        while len(samples) < batch_size:
            started = time.perf_counter()
            raw_sample = next(sample_iter, None)
            timings["load"] += time.perf_counter() - started
            if raw_sample is None:
                break

            sample_id = raw_sample["sample_id"]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Loader returned invalid sample_id: {sample_id!r}")
            if sample_id in processed_sample_ids:
                continue

            started = time.perf_counter()
            conversations.append(resolve_messages(loader, raw_sample))
            timings["encode"] += time.perf_counter() - started

            samples.append(
                {
                    "sample_id": sample_id,
                    "metadata": dict(raw_sample.get("metadata") or {}),
                    "image_bytes": bytes(raw_sample["image_bytes"]),
                }
            )

        return (samples, conversations) if samples else None

    batch = prepare_batch()
    if batch is None:
        os.replace(tmp_path, final_path)
        print(f"No remaining work for task; completed output: {final_path}")
        return

    queue: Queue[tuple[list[dict], list[list[dict]]] | Exception | object] = Queue(maxsize=prefetch_batches)
    done = object()

    def prefetch() -> None:
        try:
            while True:
                next_batch = prepare_batch()
                if next_batch is None:
                    break
                queue.put(next_batch)
        except Exception as exc:
            queue.put(exc)
        finally:
            queue.put(done)

    thread = Thread(target=prefetch, name="batch-prefetch", daemon=True)
    thread.start()

    print("Loading model...")
    if inference_backend == "vllm":
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
            max_num_seqs=batch_size,
            **({"limit_mm_per_prompt": limit_mm_per_prompt} if limit_mm_per_prompt is not None else {}),
            **({"mm_processor_kwargs": mm_processor_kwargs} if mm_processor_kwargs is not None else {}),
            **extra_llm_kwargs,
        )
        tokenizer = llm.get_tokenizer()
        processor = None
        model = None
    elif inference_backend == "transformers":
        processor, model = load_transformers_runtime(
            model_path=model_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            extra_model_kwargs=transformers_model_kwargs,
        )
        llm = None
        tokenizer = processor.tokenizer
    else:
        raise ValueError(f"Unsupported inference backend: {inference_backend}")

    processed_now = 0
    with tmp_path.open("a", encoding="utf-8") as output_handle:
        while batch is not None:
            samples, conversations = batch

            if inference_backend == "vllm":
                started = time.perf_counter()
                outputs = llm.chat(
                    conversations,
                    sampling_params=sampling,
                    use_tqdm=False,
                    chat_template_kwargs=chat_template_kwargs,
                )
                timings["model"] += time.perf_counter() - started
                normalized_outputs = [
                    {
                        "text": output.outputs[0].text.strip(),
                        "generation_seconds": resolve_generation_seconds(output),
                        "prompt_tokens": len(output.prompt_token_ids or []),
                        "completion_tokens": len(output.outputs[0].token_ids or []),
                        "time_to_first_token_seconds": getattr(
                            getattr(output, "metrics", None),
                            "first_token_latency",
                            None,
                        ),
                    }
                    for output in outputs
                ]
            else:
                normalized_outputs = run_transformers_batch(
                    samples=samples,
                    conversations=conversations,
                    processor=processor,
                    model=model,
                    sampling=sampling,
                    chat_template_kwargs=chat_template_kwargs,
                )
                timings["model"] += sum(output["generation_seconds"] for output in normalized_outputs)

            for sample, output in zip(samples, normalized_outputs):
                raw_caption = output["text"]
                caption, thinking_text = strip_thinking(raw_caption)
                if not caption:
                    raise RuntimeError(f"Model returned an empty caption for {sample['sample_id']!r}")

                thinking_tokens = count_tokens(tokenizer, thinking_text)
                caption_tokens = count_tokens(tokenizer, caption)
                payload = {
                    "sample_id": sample["sample_id"],
                    "caption": caption,
                }
                sample_metadata = dict(sample["metadata"])
                sample_metadata["generation_seconds"] = output["generation_seconds"]
                sample_metadata["time_to_first_token_seconds"] = output.get("time_to_first_token_seconds")
                sample_metadata["prompt_tokens"] = output["prompt_tokens"]
                sample_metadata["completion_tokens"] = output["completion_tokens"]
                sample_metadata["caption_tokens"] = caption_tokens
                sample_metadata["thinking_tokens"] = thinking_tokens
                sample_metadata["thinking_present"] = bool(thinking_text)
                if sample_metadata:
                    payload["metadata"] = sample_metadata

                output_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                processed_sample_ids.add(sample["sample_id"])
                processed_now += 1

            # Persist every completed batch so repairs can resume from durable JSONL.
            output_handle.flush()
            os.fsync(output_handle.fileno())

            print(f"Processed {processed_now} samples in this run")

            next_item = queue.get()
            if next_item is done:
                batch = None
            elif isinstance(next_item, Exception):
                raise next_item
            else:
                batch = next_item

    thread.join()

    total_seconds = time.perf_counter() - total_started
    if processed_now:
        print(
            "Timing summary: "
            f"load={timings['load']:.2f}s "
            f"encode={timings['encode']:.2f}s "
            f"model={timings['model']:.2f}s "
            f"total={total_seconds:.2f}s "
            f"samples_per_sec={processed_now / total_seconds:.2f}"
        )

    os.replace(tmp_path, final_path)
    print(f"Completed task output: {final_path}")


if __name__ == "__main__":
    main()
