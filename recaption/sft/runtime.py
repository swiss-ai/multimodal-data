from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from sft_recaption.config import LOGS_DIR, ModelConfig
from sft_recaption.schemas import ImagePayload


class ChatEngine(Protocol):
    def chat(
        self,
        conversations: list[list[dict[str, object]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[str]: ...


def to_data_url(image: ImagePayload) -> str:
    encoded = base64.b64encode(image.data).decode("ascii")
    return f"data:{image.media_type};base64,{encoded}"


@dataclass(slots=True)
class VLLMChatEngine:
    config: ModelConfig
    _sampling_cls: Any = field(init=False, repr=False)
    _llm: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from vllm import LLM, SamplingParams

        self._sampling_cls = SamplingParams
        self._llm = LLM(
            model=self.config.model_repo,
            tokenizer=self.config.model_repo,
            download_dir=str(self.config.download_dir),
            tensor_parallel_size=self.config.tensor_parallel_size,
            enforce_eager=self.config.enforce_eager,
            dtype=self.config.dtype,
            max_model_len=self.config.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            trust_remote_code=self.config.trust_remote_code,
            max_num_seqs=self.config.max_num_seqs,
            **(
                {"limit_mm_per_prompt": self.config.limit_mm_per_prompt}
                if self.config.limit_mm_per_prompt is not None
                else {}
            ),
            **(
                {"mm_processor_kwargs": self.config.mm_processor_kwargs}
                if self.config.mm_processor_kwargs is not None
                else {}
            ),
        )

    def chat(
        self,
        conversations: list[list[dict[str, object]]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> list[str]:
        sampling = self._sampling_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self._llm.chat(
            conversations,
            sampling_params=sampling,
            use_tqdm=False,
            chat_template_kwargs=self.config.chat_template_kwargs,
        )
        return [output.outputs[0].text.strip() for output in outputs]


def configure_worker_environment(worker_index: int) -> None:
    base_cache_root = os.environ.get("SFT_RECAPTION_CACHE_ROOT")
    if base_cache_root:
        worker_root = Path(base_cache_root) / f"w{worker_index}"
    else:
        worker_root = Path(f"/tmp/{os.environ.get('USER', 'user')}/sftrecap-w{worker_index}")
    env_map = {
        "HF_HOME": worker_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": worker_root / "hub",
        "XDG_CACHE_HOME": worker_root / "xdg-cache",
        "TRITON_CACHE_DIR": worker_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": worker_root / "torchinductor",
        "VLLM_CACHE_ROOT": worker_root / "vllm",
        "VLLM_RPC_BASE_PATH": worker_root / "vllm-rpc",
        "FLASHINFER_WORKSPACE_BASE": worker_root / "flashinfer",
    }
    for path in env_map.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    for key, path in env_map.items():
        os.environ[key] = str(path)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
