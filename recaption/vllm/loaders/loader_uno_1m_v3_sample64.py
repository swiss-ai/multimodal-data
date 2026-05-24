from itertools import islice
from pathlib import Path
from types import SimpleNamespace

from loader_uno_1m_v3 import (
    SYSTEM_PROMPT,
    build_prompt,
)
from loader_uno_1m_v3 import (
    iter_samples as base_iter_samples,
)
from loader_uno_1m_v3 import (
    loader as base_loader,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "uno_1m_v3_sample64"


def iter_samples(task_id: int, task_count: int):
    yield from islice(base_iter_samples(task_id, task_count), 64)


loader = SimpleNamespace(
    name="uno_1m_v3_sample64",
    output_dir=OUTPUT_DIR,
    build_prompt=build_prompt,
    system_prompt=SYSTEM_PROMPT,
    iter_samples=iter_samples,
    batch_size=base_loader.batch_size,
    sampling_params=base_loader.sampling_params,
)
