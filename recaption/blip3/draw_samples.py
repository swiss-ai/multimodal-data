#!/usr/bin/env python3
"""Draw bounding boxes on samples from a recap tar and save annotated images + captions."""

import io
import json
import re
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

TAR_PATH = Path("/path/to/data/vision-datasets/raw/stage2/hf___Salesforce___blip3-grounding-50m___recap/0000_00.tar")
OUTPUT_DIR = Path("/path/to/data/vision-datasets/raw/stage2/hf___Salesforce___blip3-grounding-50m___recap/sample")
N_SAMPLES = 50

COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (67, 99, 216),
    (245, 130, 49),
    (145, 30, 180),
    (66, 212, 244),
    (240, 50, 230),
    (191, 239, 69),
    (70, 153, 144),
    (220, 190, 255),
    (154, 99, 36),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (0, 0, 117),
]

_OBJECT_RE = re.compile(r"<object>([^<]*)</object>")
_BBOX_COORDS_RE = re.compile(r"<bbox>\[([^\]]+)\]")


def parse_objects(caption: str) -> list[tuple[str, list[list[float]]]]:
    objects = list(_OBJECT_RE.finditer(caption))
    result = []
    for i, obj in enumerate(objects):
        seg_end = objects[i + 1].start() if i + 1 < len(objects) else len(caption)
        segment = caption[obj.end() : seg_end]
        bboxes = []
        for coords_str in _BBOX_COORDS_RE.findall(segment):
            try:
                coords = [float(x.strip()) for x in coords_str.split(",")]
                if len(coords) == 4:
                    bboxes.append(coords)
            except ValueError:
                pass
        if bboxes:
            result.append((obj.group(1).strip(), bboxes))
    return result


def draw_and_save(img: Image.Image, caption: str, out_stem: Path):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=max(12, h // 60),
        )
    except Exception:
        font = ImageFont.load_default()

    for idx, (label, bboxes) in enumerate(parse_objects(caption)):
        color = COLORS[idx % len(COLORS)]
        for x0, y0, x1, y1 in bboxes:
            px0 = int(x0 / 1000 * w)
            py0 = int(y0 / 1000 * h)
            px1 = int(x1 / 1000 * w)
            py1 = int(y1 / 1000 * h)
            for t in range(2):
                draw.rectangle([px0 - t, py0 - t, px1 + t, py1 + t], outline=color)
            tw = draw.textlength(label, font=font)
            th = font.size if hasattr(font, "size") else 12
            draw.rectangle([px0, py0 - th - 4, px0 + tw + 4, py0], fill=color)
            draw.text((px0 + 2, py0 - th - 2), label, fill=(255, 255, 255), font=font)

    img.save(str(out_stem) + ".png")
    out_stem.with_suffix(".txt").write_text(caption)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(TAR_PATH, "r") as tf:
        members = {m.name: m for m in tf.getmembers()}
        keys = sorted(k[:-4] for k in members if k.endswith(".jpg"))[:N_SAMPLES]

        for i, key in enumerate(keys):
            img = Image.open(io.BytesIO(tf.extractfile(members[f"{key}.jpg"]).read())).convert("RGB")
            record = json.loads(tf.extractfile(members[f"{key}.json"]).read())
            caption = record["caption"]

            out_stem = OUTPUT_DIR / f"{i:02d}_{key}"
            draw_and_save(img, caption, out_stem)
            print(f"[{i + 1}/{N_SAMPLES}] {key} — {len(parse_objects(caption))} objects")

    print(f"\nDone. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
