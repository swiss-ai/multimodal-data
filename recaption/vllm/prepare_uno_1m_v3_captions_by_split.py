#!/usr/bin/env python3

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

CAPTIONS_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "uno_1m_v3"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "cache" / "uno_1m_v3_by_split"
PROGRESS_EVERY = 10_000


def normalize_caption(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\r\n", "\n").strip()


def split_name_for_sample_id(sample_id: str) -> str:
    head, _, _ = sample_id.partition("/")
    if not head.startswith("split"):
        raise ValueError(f"Unexpected UNO sample id {sample_id!r}")
    return head


def main() -> None:
    output_root = DEFAULT_OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    caption_paths = sorted(CAPTIONS_ROOT.glob("captions_task*.jsonl"))
    if not caption_paths:
        raise FileNotFoundError(f"No UNO v3 caption files found under {CAPTIONS_ROOT}")

    stats_by_split: dict[str, int] = {}
    loaded = 0

    with ExitStack() as stack:
        handles: dict[str, object] = {}

        for caption_path in caption_paths:
            with caption_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    payload = json.loads(line)
                    sample_id = payload.get("sample_id")
                    caption = normalize_caption(payload.get("caption"))
                    metadata = dict(payload.get("metadata") or {})
                    if not sample_id or not caption:
                        continue

                    split_name = split_name_for_sample_id(str(sample_id))
                    out_handle = handles.get(split_name)
                    if out_handle is None:
                        out_path = output_root / f"{split_name}.jsonl"
                        out_handle = stack.enter_context(out_path.open("w", encoding="utf-8"))
                        handles[split_name] = out_handle

                    out_handle.write(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "caption": caption,
                                "metadata": metadata,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    stats_by_split[split_name] = stats_by_split.get(split_name, 0) + 1
                    loaded += 1
                    if loaded % PROGRESS_EVERY == 0:
                        print(f"Prepared {loaded} UNO captions", flush=True)

    summary = {
        "captions_prepared": loaded,
        "split_count": len(stats_by_split),
        "counts_by_split": dict(sorted(stats_by_split.items())),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
