"""
Normalize pixmo-point-explanations coordinates from 0-100 percentage range to 0-1.

Coordinate format in assistant messages:
  Single point:   [x, y]           -> [x/100, y/100]
  Multiple points: [[x1,y1],[x2,y2]] -> [[x1/100,y1/100],[x2/100,y2/100]]
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(
    os.path.join(
        os.environ.get("DATA_ROOT", "./data"),
        "vision-datasets/raw/sft/hf___allenai___pixmo-point-explanations/preprocessed",
    )
)
DST = Path(
    os.path.join(
        os.environ.get("DATA_ROOT", "./data"),
        "vision-datasets/raw/sft/hf___allenai___pixmo-point-explanations/preprocessed2",
    )
)

# Matches both:
#   [76.8, 63.8]
#   [[43.1, 71.4], [40.9, 71.8]]
COORD_RE = re.compile(r"^\[.*\]$", re.DOTALL)


def is_coordinate_content(content: str) -> bool:
    content = content.strip()
    return bool(COORD_RE.match(content))


def normalize_coord_string(content: str) -> str:
    parsed = json.loads(content.strip())
    # Single point: [x, y]
    if isinstance(parsed[0], (int, float)):
        x, y = parsed
        return json.dumps([round(x / 100, 6), round(y / 100, 6)])
    # Multiple points: [[x1,y1], [x2,y2], ...]
    normalized = [[round(pt[0] / 100, 6), round(pt[1] / 100, 6)] for pt in parsed]
    return json.dumps(normalized)


def normalize_messages(messages: np.ndarray) -> np.ndarray:
    result = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant" and is_coordinate_content(content):
            content = normalize_coord_string(content)
        result.append({"role": role, "content": content})
    return np.array(result, dtype=object)


def process_file(src_path: Path, dst_path: Path) -> None:
    df = pd.read_parquet(src_path)
    df["messages"] = df["messages"].apply(normalize_messages)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)
    print(f"  {src_path.name} -> {dst_path.name}  ({len(df)} rows)")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    parquets = sorted(SRC.glob("*.parquet"))
    print(f"Processing {len(parquets)} files...")
    for src in parquets:
        process_file(src, DST / src.name)
    print("Done.")


if __name__ == "__main__":
    main()
