"""Export best-prompt captions into a README for side-by-side review."""
import re
import shutil
from pathlib import Path

SRC = Path(__file__).parent / "outputs"
DST = Path("/capstor/scratch/cscs/xyixuan/recon_examples/waffle_caption_ablation")
DST.mkdir(parents=True, exist_ok=True)

for p in list(DST.glob("sample_*.jpg")) + list(DST.glob("s*.txt")) + \
         list(DST.glob("index.html")) + list(DST.glob("README.md")):
    p.unlink()


def parse(path: Path) -> dict:
    text = path.read_text()
    out = {"persona": "", "building": "", "dt": "", "words": "", "caption": text}
    pm = re.search(r"^PERSONA: (.*)", text, re.MULTILINE)
    if pm:
        out["persona"] = pm.group(1).strip()
    bm = re.search(r"^BUILDING [^:]+: (.*)", text, re.MULTILINE)
    if bm:
        out["building"] = bm.group(1).strip()
    cm = re.search(r"CAPTION \(([\d.]+)s, (\d+) words\):\n(.*)", text, re.DOTALL)
    if cm:
        out["dt"] = cm.group(1)
        out["words"] = cm.group(2)
        out["caption"] = cm.group(3).strip()
    return out


images = sorted(SRC.glob("sample_*.jpg"))
for img in images:
    shutil.copy(img, DST / img.name)

lines = [
    "# WAFFLE Captions — Qwen3.5-397B-A17B (production prompt)",
    "",
    "Single consolidated prompt, no metadata in user message, persona calibrated "
    "via `high_level_building_type`. Caption-only output (no two-section format). "
    "Mandatory full transcription of enumerated lists. `enable_thinking: False`.",
    "",
    f"**Samples:** {len(images)}  ·  **Target length:** 400-800 words  ·  "
    f"**MAX_TOKENS:** 2400  ·  **Concurrency:** 8",
    "",
]

for img in images:
    idx = img.stem.split("_")[1]
    txt = SRC / f"s{int(idx):02d}.txt"
    if not txt.exists():
        continue
    p = parse(txt)
    lines.append("---")
    lines.append(f"## Sample {idx}")
    if p["building"]:
        lines.append(f"_{p['building']}_")
    lines.append("")
    lines.append(f"![sample {idx}](sample_{idx}.jpg)")
    lines.append("")
    lines.append(f"**Persona:** {p['persona']}  ·  "
                 f"**{p['dt']}s**  ·  **{p['words']} words**")
    lines.append("")
    for para in p["caption"].split("\n"):
        lines.append(f"> {para}" if para.strip() else ">")
    lines.append("")

(DST / "README.md").write_text("\n".join(lines) + "\n")
print(f"wrote {DST}/README.md  —  {len(images)} samples")
