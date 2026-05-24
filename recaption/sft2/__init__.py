from sft_recaption.loaders.base import BaseLoader
from sft_recaption.loaders.drim_visual_reason_hard import DrimVisualReasonHardLoader
from sft_recaption.loaders.mint_1t_arxiv_processed import Mint1TArxivProcessedLoader

LOADER_REGISTRY: dict[str, type[BaseLoader]] = {
    "drim_visual_reason_hard": DrimVisualReasonHardLoader,
    "mint_1t_arxiv_processed": Mint1TArxivProcessedLoader,
}


def create_loader(name: str) -> BaseLoader:
    try:
        loader_cls = LOADER_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown loader {name!r}; expected one of {sorted(LOADER_REGISTRY)}") from exc
    return loader_cls()
