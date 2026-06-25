"""Convert cyberagent/crello rows into BLIP3-Grounding-style text using
ground-truth designer-authored coordinates (no Qwen needed).

Each row's `left/top/width/height` per element are already pixel-perfect
ground truth. We normalise them to [0.0, 1.0] of the canvas and emit the
BLIP3-Grounding Level-2 schema:

    This is a {format} design titled "{title}".
    The design contains <object>{semantic}</object><bbox>[x1,y1][x2,y2]</bbox>,
    <object>{semantic}</object><bbox>[x1,y1][x2,y2]</bbox>, ...

Output schema (per source parquet shard, one row per design):
    id              : design id
    title           : original title
    canvas_width    : int
    canvas_height   : int
    n_elements      : int
    image           : preview JPEG bytes
    grounded_text   : BLIP3-style description using ground-truth coords
"""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Crello element type codes — inferred from the schema. Not officially
# documented per code; common pattern is image=0/3, text=1, shape=2.
# We treat anything with non-empty `text` as a text element regardless.

# Format codes from HF dataset card examples — partial mapping. Unmapped
# codes are emitted as numeric "format-XX" strings in the prose preamble.
FORMAT_NAMES = {
    0: "Instagram story",
    7: "Instagram square",
    24: "vertical web banner",
    35: "Twitter post",
    36: "Facebook cover",
}


def normalise_bbox(left: float, top: float, w: float, h: float,
                   cw: int, ch: int) -> tuple[float, float, float, float]:
    """Convert pixel left/top/width/height to normalised [x1, y1, x2, y2]."""
    if cw <= 0 or ch <= 0:
        return 0.0, 0.0, 0.0, 0.0
    x1 = max(0.0, left / cw)
    y1 = max(0.0, top / ch)
    x2 = min(1.0, (left + w) / cw)
    y2 = min(1.0, (top + h) / ch)
    return x1, y1, x2, y2


def truncate_text(s: str, max_len: int = 60) -> str:
    s = (s or "").strip().replace("\n", " ")
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def element_label(idx: int, type_code: int, text: str) -> str:
    """Pick a short semantic-ish label for an element."""
    text = (text or "").strip()
    if text:
        return f'text "{truncate_text(text, 60)}"'
    # No text → image or shape. Crello doesn't tell us which precisely
    # without inspecting the image bytes, so generic.
    if type_code == 0 or type_code == 3:
        return "image element"
    if type_code == 1:
        return "decorative shape"
    return "graphic element"


def format_label(format_code: int) -> str:
    return FORMAT_NAMES.get(format_code, f"design (format-{format_code})")


def build_grounded_text(row: dict) -> str:
    cw = int(row.get("canvas_width") or 0)
    ch = int(row.get("canvas_height") or 0)
    title = (row.get("title") or "").strip()
    fmt = format_label(row.get("format") or 0)
    n_elem = int(row.get("length") or 0)

    types = row.get("type") or []
    lefts = row.get("left") or []
    tops = row.get("top") or []
    widths = row.get("width") or []
    heights = row.get("height") or []
    texts = row.get("text") or []

    parts = []
    for i in range(n_elem):
        try:
            l, t, w, h = lefts[i], tops[i], widths[i], heights[i]
            tcode = int(types[i])
            etext = texts[i] if i < len(texts) else ""
        except (IndexError, TypeError):
            continue
        x1, y1, x2, y2 = normalise_bbox(l, t, w, h, cw, ch)
        # Skip degenerate (zero-area) elements
        if x2 <= x1 + 1e-3 or y2 <= y1 + 1e-3:
            continue
        label = element_label(i, tcode, etext)
        parts.append(
            f"<object>{label}</object>"
            f"<bbox>[{x1:.3f}, {y1:.3f}][{x2:.3f}, {y2:.3f}]</bbox>"
        )

    if not parts:
        return ""

    intro_bits = [f"This is a {fmt}"]
    if title:
        intro_bits.append(f'titled "{truncate_text(title, 80)}"')
    intro_bits.append(f"on a {cw}x{ch} canvas with {n_elem} elements.")
    intro = " ".join(intro_bits[:1] + intro_bits[1:]).rstrip(",")
    if not intro.endswith("."):
        intro += "."

    return intro + "\nThe design contains " + ", ".join(parts) + "."


def run_shard(shard: Path, out_dir: Path):
    out_path = out_dir / shard.name
    if out_path.exists():
        print(f"[skip] {shard.name}")
        return
    cols = ["id", "preview", "canvas_width", "canvas_height", "title",
            "format", "category", "length",
            "type", "left", "top", "width", "height", "text"]
    table = pq.read_table(shard, columns=cols)
    rows = table.to_pylist()
    out_rows = []
    skipped = 0
    for r in rows:
        prev = r.get("preview") or {}
        img_bytes = prev.get("bytes") if isinstance(prev, dict) else None
        if not img_bytes:
            skipped += 1; continue
        text = build_grounded_text(r)
        if not text:
            skipped += 1; continue
        out_rows.append({
            "id":            r["id"],
            "title":         r.get("title") or "",
            "canvas_width":  int(r.get("canvas_width") or 0),
            "canvas_height": int(r.get("canvas_height") or 0),
            "n_elements":    int(r.get("length") or 0),
            "image":         img_bytes,
            "grounded_text": text,
        })
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("canvas_width", pa.int64()),
        pa.field("canvas_height", pa.int64()),
        pa.field("n_elements", pa.int64()),
        pa.field("image", pa.binary()),
        pa.field("grounded_text", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(out_rows, schema=schema), out_path,
                   compression="zstd")
    print(f"[{shard.name}] wrote {len(out_rows):,} rows ({skipped} skipped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/stage2/hf___cyberagent___crello/data")
    ap.add_argument("--output-dir", default="/capstor/scratch/cscs/xyixuan/recon/crello_grounded")
    ap.add_argument("--shards", nargs="*")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(Path(args.input_dir).glob("*.parquet"))
    if args.shards:
        shards = [s for s in shards if s.name in args.shards]
    print(f"will process {len(shards)} shards")
    for s in shards:
        run_shard(s, out_dir)
    print("done.")


if __name__ == "__main__":
    main()
