from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = Path(os.environ.get("SFT_RECAPTION_ARTIFACTS_DIR", str(WORKDIR / "artifacts")))
MANIFESTS_DIR = ARTIFACTS_DIR / "manifests"
CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"
CURATED_DIR = ARTIFACTS_DIR / "curated"
PARQUET_EXPORT_DIR = ARTIFACTS_DIR / "parquet"
CACHE_DIR = Path(os.environ.get("SFT_RECAPTION_CACHE_DIR", str(WORKDIR / ".cache")))
LOGS_DIR = Path(os.environ.get("SFT_RECAPTION_LOGS_DIR", str(WORKDIR / "logs")))
SHARED_MODEL_CACHE_ROOT = Path("/tmp/models")

DEFAULT_MODEL_REPO = "google/gemma-4-26B-A4B-it"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.80
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_MAX_NUM_SEQS = 8
DEFAULT_LIMIT_MM_PER_PROMPT = {"image": 4}
DEFAULT_CHAT_TEMPLATE_KWARGS = {}
DEFAULT_CANDIDATES_PER_SAMPLE = 2
DEFAULT_TASK_COUNT = 4
GENERATION_PROMPT_VERSION = "gemma4_reasoning_qa_v3"
JUDGE_PROMPT_VERSION = "gemma4_judge_v1"


def shared_model_cache_root() -> Path:
    return Path(os.environ.get("SFT_RECAPTION_MODEL_CACHE_ROOT", str(SHARED_MODEL_CACHE_ROOT)))


def resolve_model_reference(model_ref: str) -> str:
    explicit_path = os.environ.get("SFT_RECAPTION_MODEL_PATH")
    if explicit_path:
        path = Path(explicit_path)
        if path.exists():
            return str(path)

    candidate = Path(model_ref)
    if candidate.exists():
        return str(candidate)

    if model_ref != DEFAULT_MODEL_REPO:
        return model_ref

    cache_root = shared_model_cache_root()
    snapshot_root = cache_root / "models--google--gemma-4-26B-A4B-it" / "snapshots"
    if not snapshot_root.exists():
        return model_ref

    snapshots = sorted(path for path in snapshot_root.iterdir() if path.is_dir())
    if not snapshots:
        return model_ref
    return str(snapshots[-1])


def resolve_model_download_dir() -> Path:
    cache_root = shared_model_cache_root()
    if cache_root.exists():
        return cache_root
    return CACHE_DIR / "models"


def default_gpu_memory_utilization() -> float:
    return float(
        os.environ.get(
            "SFT_RECAPTION_GPU_MEMORY_UTILIZATION",
            str(DEFAULT_GPU_MEMORY_UTILIZATION),
        )
    )


def default_chat_template_kwargs() -> dict[str, object]:
    enable_thinking = os.environ.get("SFT_RECAPTION_ENABLE_THINKING")
    if enable_thinking is None:
        return dict(DEFAULT_CHAT_TEMPLATE_KWARGS)
    return {"enable_thinking": enable_thinking == "1"}


@dataclass(slots=True)
class ModelConfig:
    model_repo: str = DEFAULT_MODEL_REPO
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    dtype: str = DEFAULT_DTYPE
    gpu_memory_utilization: float = field(default_factory=default_gpu_memory_utilization)
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    limit_mm_per_prompt: dict[str, int] | None = field(default_factory=lambda: dict(DEFAULT_LIMIT_MM_PER_PROMPT))
    mm_processor_kwargs: dict[str, int] | None = None
    chat_template_kwargs: dict[str, object] = field(default_factory=default_chat_template_kwargs)
    trust_remote_code: bool = True
    download_dir: Path = field(default_factory=resolve_model_download_dir)


def ensure_runtime_dirs() -> None:
    for path in (
        ARTIFACTS_DIR,
        MANIFESTS_DIR,
        CANDIDATES_DIR,
        CURATED_DIR,
        PARQUET_EXPORT_DIR,
        CACHE_DIR,
        LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
