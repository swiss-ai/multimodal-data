import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

from vllm import SamplingParams

WORKDIR = Path(__file__).resolve().parent
MODEL_DIR = "/tmp/models"
MODEL_REPO = "utter-project/EuroLLM-22B-Instruct-2512"
MODEL_CACHE_DIR = Path(MODEL_DIR) / "models--utter-project--EuroLLM-22B-Instruct-2512"
ENGLISH_CAPTIONS_DIR = WORKDIR / "outputs" / "mitsua_art_museums_pd_440k" / "en"
LANGUAGE_CONFIGS = {
    "ro": {"name": "Romanian", "output_subdir": "ro"},
    "sv": {"name": "Swedish", "output_subdir": "sv"},
    "da": {"name": "Danish", "output_subdir": "da"},
    "fi": {"name": "Finnish", "output_subdir": "fi"},
    "bg": {"name": "Bulgarian", "output_subdir": "bg"},
    "cs": {"name": "Czech", "output_subdir": "cs"},
    "sk": {"name": "Slovak", "output_subdir": "sk"},
    "sl": {"name": "Slovenian", "output_subdir": "sl"},
    "hu": {"name": "Hungarian", "output_subdir": "hu"},
    "el": {"name": "Greek", "output_subdir": "el"},
    "uk": {"name": "Ukrainian", "output_subdir": "uk"},
    "hr": {"name": "Croatian", "output_subdir": "hr"},
}
TARGET_LANG = os.environ["RECAPTION_TARGET_LANG"]
try:
    TARGET_CONFIG = LANGUAGE_CONFIGS[TARGET_LANG]
except KeyError as exc:
    raise KeyError(
        f"Unsupported RECAPTION_TARGET_LANG={TARGET_LANG!r}; expected one of {sorted(LANGUAGE_CONFIGS)}"
    ) from exc

OUTPUT_DIR = WORKDIR / "outputs" / "mitsua_art_museums_pd_440k" / TARGET_CONFIG["output_subdir"]
SYSTEM_PROMPT = (
    "You are EuroLLM, a multilingual assistant specialized in European languages. "
    "Write accurate, natural museum descriptions using only the supplied English "
    "museum description as evidence, and return only the final text in the target language."
)
PROMPT_TEMPLATE = """Write an extensive museum description in {language_name} in 2 to 4 sentences.
Use only the English museum description below as evidence.
Preserve the factual content about the object or scene, medium, materials, composition, subject, style, display context, and visible age or wear that the English text already contains.
Rewrite it as natural, idiomatic {language_name} prose rather than a literal sentence-by-sentence translation.
Keep the tone descriptive and curatorial.
Do not add new visual details, dates, provenance, symbolism, artist biography, or any other claims not supported by the English description.
Do not mention the English source text, the prompt, uncertainty, or your reasoning.
Return only the final description in {language_name}.

English museum description:
{caption_en}
"""


def iter_english_caption_paths() -> list[Path]:
    caption_paths = sorted(
        path for path in ENGLISH_CAPTIONS_DIR.glob("captions_task*.jsonl") if path.stat().st_size > 0
    )
    if not caption_paths:
        raise FileNotFoundError(f"No English caption files found under {ENGLISH_CAPTIONS_DIR}")
    return caption_paths


def normalize_caption(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def build_messages(sample: dict) -> list[dict]:
    caption_en = normalize_caption(sample.get("conditioning_caption_en"))
    if not caption_en:
        raise ValueError(f"Sample {sample['sample_id']!r} is missing the English recap caption")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                language_name=TARGET_CONFIG["name"],
                caption_en=caption_en,
            ),
        },
    ]


def iter_samples_from_caption_file(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            caption_en = normalize_caption(payload.get("caption"))
            metadata = payload.get("metadata") or {}
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Invalid sample_id in {path}:{line_number}")
            if not caption_en:
                raise ValueError(f"Missing English caption for sample_id={sample_id!r} in {path}:{line_number}")
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid metadata for sample_id={sample_id!r} in {path}:{line_number}")

            yield {
                "sample_id": sample_id,
                "conditioning_caption_en": caption_en,
                "metadata": metadata,
            }


def iter_samples(task_id: int, task_count: int) -> Iterator[dict]:
    caption_paths = iter_english_caption_paths()
    total = len(caption_paths)
    assigned_paths = caption_paths[total * task_id // task_count : total * (task_id + 1) // task_count]
    print(
        f"Discovered {total} English caption shards under {ENGLISH_CAPTIONS_DIR}; "
        f"task {task_id}/{task_count - 1} assigned {len(assigned_paths)}"
    )
    if assigned_paths:
        print(f"Assigned first caption shard: {assigned_paths[0]}")
        print(f"Assigned last caption shard:  {assigned_paths[-1]}")

    for path in assigned_paths:
        yield from iter_samples_from_caption_file(path)


loader = SimpleNamespace(
    name=f"mitsua_art_museums_pd_440k_eurollm_from_en___{TARGET_LANG}",
    output_dir=OUTPUT_DIR,
    build_messages=build_messages,
    iter_samples=iter_samples,
    model_dir=MODEL_DIR,
    model_repo=MODEL_REPO,
    model_cache_dir=MODEL_CACHE_DIR,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096,
    batch_size=64,
    prefetch_batches=1,
    max_num_batched_tokens=4096,
    limit_mm_per_prompt=None,
    mm_processor_kwargs=None,
    chat_template_kwargs={},
    sampling_params=SamplingParams(
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        max_tokens=320,
    ),
)
