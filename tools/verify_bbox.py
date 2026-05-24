"""
Draw coordinate points on 10 sample images to verify normalization.
Saves annotated images to data/samples/.
"""

import io
import json
import os
import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageColor, ImageDraw

SRC = Path(
    os.path.join(
        os.environ.get("DATA_ROOT", "./data"),
        "vision-datasets/raw/sft/hf___allenai___pixmo-point-explanations/preprocessed",
    )
)
DST_NORM = Path(
    os.path.join(
        os.environ.get("DATA_ROOT", "./data"),
        "vision-datasets/raw/sft/hf___allenai___pixmo-point-explanations/preprocessed2",
    )
)
OUT = Path(os.path.join(os.environ.get("SCRATCH_DIR", "/tmp"), "toolbox/bbox/data/samples"))
OUT.mkdir(parents=True, exist_ok=True)

COORD_RE = re.compile(r"^\[.*\]$", re.DOTALL)
RADIUS = 8
COLORS = ["red", "blue", "green", "orange", "magenta", "cyan", "yellow", "lime"]


def is_coord(content: str) -> bool:
    return bool(COORD_RE.match(content.strip()))


def parse_coords(content: str):
    parsed = json.loads(content.strip())
    if isinstance(parsed[0], (int, float)):
        return [parsed]
    return parsed


def annotate(img: Image.Image, norm_points_list) -> Image.Image:
    img = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size

    point_idx = 0
    for group_idx, pts in enumerate(norm_points_list):
        color = COLORS[group_idx % len(COLORS)]
        rgb = ImageColor.getrgb(color)
        for nx, ny in pts:
            px = nx * w
            py = ny * h
            draw.ellipse(
                [px - RADIUS, py - RADIUS, px + RADIUS, py + RADIUS],
                fill=(*rgb, 200),
                outline="white",
                width=2,
            )
            draw.text((px + RADIUS + 2, py - RADIUS), str(point_idx), fill="white")
            point_idx += 1

    return Image.alpha_composite(img, overlay).convert("RGB")


def main():
    df_src = pd.read_parquet(SRC / "0000.parquet")
    df_norm = pd.read_parquet(DST_NORM / "0000.parquet")

    saved = 0
    for row_idx in range(len(df_src)):
        if saved >= 10:
            break

        src_row = df_src.iloc[row_idx]
        norm_row = df_norm.iloc[row_idx]

        norm_groups = []
        orig_groups = []

        for src_msg, norm_msg in zip(src_row["messages"], norm_row["messages"]):
            if norm_msg["role"] == "assistant" and is_coord(norm_msg["content"]):
                norm_groups.append(parse_coords(norm_msg["content"]))
                orig_groups.append(parse_coords(src_msg["content"]))

        if not norm_groups:
            continue

        img = Image.open(io.BytesIO(src_row["jpg"]))
        w, h = img.size

        annotated = annotate(img, norm_groups)

        # Build point index so we can annotate which pt# maps to which message
        point_map = {}  # (group_idx, pt_in_group) -> global pt index
        pidx = 0
        for gi, ng in enumerate(norm_groups):
            for pi in range(len(ng)):
                point_map[(gi, pi)] = pidx
                pidx += 1

        # Write plaintext transcript
        txt_lines = [f"image: {w}x{h}  url: {src_row['image_url']}", ""]
        group_idx = 0
        for src_msg, norm_msg in zip(src_row["messages"], norm_row["messages"]):
            role = src_msg["role"]
            content = src_msg["content"]
            if role == "assistant" and is_coord(norm_msg["content"]):
                ng = norm_groups[group_idx]
                pt_labels = ", ".join(
                    f"pt{point_map[(group_idx, pi)]}=({ng[pi][0]:.4f},{ng[pi][1]:.4f})" for pi in range(len(ng))
                )
                txt_lines.append(f"assistant: {content}  [{pt_labels}]")
                group_idx += 1
            else:
                label = "assistant" if str(role).startswith("a") else "user"
                txt_lines.append(f"{label}: {content}")
            txt_lines.append("")

        out_path = OUT / f"sample_{saved:02d}.jpg"
        txt_path = OUT / f"sample_{saved:02d}.txt"
        annotated.save(out_path, quality=90)
        txt_path.write_text("\n".join(txt_lines))
        print(f"  Saved: {out_path} + {txt_path.name}")
        saved += 1

    print(f"\nSaved {saved} samples to {OUT}")


if __name__ == "__main__":
    main()
