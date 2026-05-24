"""
DailyMed SPL: add a `markdown` column to the parquet shards by converting the
HL7 v3 SPL XML into a section-aware Markdown rendering with image anchors.

Image anchors look like:  [[IMAGE: <obs_media_id> | <filename>]]
so a downstream LLM step can interleave the actual image bytes (from the
`images` column, matched by filename) at the correct position in the prose.

Reads:  SRC_DIR/part-*.parquet  (columns: id, xml, images)
Writes: DST_DIR/part-*.parquet  (columns: id, xml, images, markdown)

Run interactively:
    .venv/bin/python markdown.py
"""

import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


SRC_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/parquet"
DST_DIR = "/path/to/data/medical-datasets/raw/dailymed_spl/parquet_md"
NUM_WORKERS = 256
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3
ROW_GROUP_ROWS = 256
CHUNK_ROWS = 4096  # rows per pool dispatch chunk

NS = "{urn:hl7-org:v3}"


def _local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _text(s):
    return s if s is not None else ""


def _collapse_ws(s: str) -> str:
    return re.sub(r"[ \t\r\n]+", " ", s).strip()


def _inline(elem) -> str:
    """Render an element's children as inline text (recursively)."""
    parts = [_text(elem.text)]
    for child in elem:
        tag = _local(child.tag)
        if tag == "br":
            parts.append("\n")
        elif tag == "content":
            inner = _inline(child)
            style = (child.get("styleCode") or "").lower()
            if "bold" in style and "italics" in style:
                parts.append(f"***{inner}***")
            elif "bold" in style:
                parts.append(f"**{inner}**")
            elif "italics" in style or "italic" in style:
                parts.append(f"*{inner}*")
            else:
                parts.append(inner)
        elif tag == "linkHtml":
            href = child.get("href") or ""
            parts.append(f"[{_inline(child)}]({href})")
        elif tag == "sup":
            parts.append(f"^{_inline(child)}^")
        elif tag == "sub":
            parts.append(f"~{_inline(child)}~")
        elif tag == "renderMultiMedia":
            ref = child.get("referencedObject") or ""
            parts.append(f"[[IMAGE: {ref}]]")
        elif tag == "footnote":
            parts.append(f"[footnote: {_inline(child)}]")
        elif tag == "footnoteRef":
            parts.append(f"[^{child.get('IDREF', '')}]")
        else:
            parts.append(_inline(child))
        parts.append(_text(child.tail))
    return "".join(parts)


def _render_list(elem, depth: int = 0) -> str:
    list_type = (elem.get("listType") or "unordered").lower()
    ordered = list_type == "ordered"
    out = []
    for i, item in enumerate(elem.findall(f"{NS}item"), 1):
        bullet = f"{i}." if ordered else "-"
        # item may have <caption> followed by text
        caption_el = item.find(f"{NS}caption")
        if caption_el is not None:
            cap = _inline(caption_el)
            tail_text = _text(caption_el.tail)
            # render rest of item content (excluding caption)
            rest_parts = [tail_text]
            for child in item:
                if child is caption_el:
                    continue
                rest_parts.append(_inline(child))
                rest_parts.append(_text(child.tail))
            rest = _collapse_ws("".join(rest_parts))
            line = f"{cap} {rest}".strip() if cap else rest
        else:
            line = _collapse_ws(_inline(item))
        out.append(f"{'  ' * depth}{bullet} {line}")
        # nested lists
        for sub in item.findall(f"{NS}list"):
            out.append(_render_list(sub, depth + 1))
    return "\n".join(out)


def _render_table(elem) -> str:
    def cells(row, tag):
        return [_collapse_ws(_inline(c)) for c in row.findall(f"{NS}{tag}")]

    head_rows = []
    for thead in elem.findall(f"{NS}thead"):
        for tr in thead.findall(f"{NS}tr"):
            head_rows.append(cells(tr, "th") or cells(tr, "td"))
    body_rows = []
    for tbody in elem.findall(f"{NS}tbody"):
        for tr in tbody.findall(f"{NS}tr"):
            body_rows.append(cells(tr, "td") or cells(tr, "th"))
    # tables sometimes have <tr> directly under <table>
    for tr in elem.findall(f"{NS}tr"):
        body_rows.append(cells(tr, "td") or cells(tr, "th"))

    if not head_rows and not body_rows:
        return ""

    ncols = max(
        (len(r) for r in head_rows + body_rows if r),
        default=1,
    )

    if not head_rows:
        head_rows = [[""] * ncols]

    def fmt_row(r):
        r = list(r) + [""] * (ncols - len(r))
        return "| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in r) + " |"

    lines = [fmt_row(head_rows[0]), "| " + " | ".join(["---"] * ncols) + " |"]
    for r in head_rows[1:]:
        lines.append(fmt_row(r))
    for r in body_rows:
        lines.append(fmt_row(r))
    return "\n".join(lines)


def _render_text_block(text_el) -> str:
    """Render a <text> element's children as block-level Markdown."""
    blocks = []
    if text_el is None:
        return ""
    # Capture leading text
    leading = _text(text_el.text)
    if leading.strip():
        blocks.append(_collapse_ws(leading))
    for child in text_el:
        tag = _local(child.tag)
        if tag == "paragraph":
            s = _collapse_ws(_inline(child))
            if s:
                blocks.append(s)
        elif tag == "list":
            s = _render_list(child)
            if s:
                blocks.append(s)
        elif tag == "table":
            s = _render_table(child)
            if s:
                blocks.append(s)
        elif tag == "renderMultiMedia":
            ref = child.get("referencedObject") or ""
            blocks.append(f"[[IMAGE: {ref}]]")
        elif tag == "br":
            pass
        else:
            s = _collapse_ws(_inline(child))
            if s:
                blocks.append(s)
        # tails between block siblings
        tail = _text(child.tail)
        if tail.strip():
            blocks.append(_collapse_ws(tail))
    return "\n\n".join(blocks)


def _media_map(root) -> dict:
    """Map observationMedia ID -> referenced filename."""
    out = {}
    for om in root.iter(f"{NS}observationMedia"):
        mid = om.get("ID") or ""
        ref_el = om.find(f"{NS}value/{NS}reference")
        if ref_el is not None and mid:
            out[mid] = ref_el.get("value") or ""
    return out


def _resolve_image_anchors(md: str, media: dict) -> str:
    def repl(m):
        mid = m.group(1)
        fname = media.get(mid, "")
        return f"[[IMAGE: {mid} | {fname}]]" if fname else f"[[IMAGE: {mid}]]"

    return re.sub(r"\[\[IMAGE: ([^\]|]+)\]\]", repl, md)


def _render_section(sec, depth: int = 2) -> str:
    """Render a <section> as Markdown. depth = heading level (## = 2)."""
    code_el = sec.find(f"{NS}code")
    display = (code_el.get("displayName") if code_el is not None else "") or ""
    title_el = sec.find(f"{NS}title")
    title = _collapse_ws(_inline(title_el)) if title_el is not None else ""

    heading_bits = []
    if display:
        heading_bits.append(display.strip())
    if title and title.lower() != display.strip().lower():
        heading_bits.append(title)
    heading = " — ".join(heading_bits) if heading_bits else ""

    parts = []
    if heading:
        parts.append(f"{'#' * min(depth, 6)} {heading}")

    text_el = sec.find(f"{NS}text")
    if text_el is not None:
        body = _render_text_block(text_el)
        if body:
            parts.append(body)

    # nested sections via <component><section>
    for comp in sec.findall(f"{NS}component"):
        for sub in comp.findall(f"{NS}section"):
            parts.append(_render_section(sub, depth + 1))

    return "\n\n".join(p for p in parts if p)


def _render_header(root) -> str:
    lines = []
    title_el = root.find(f"{NS}title")
    title = _collapse_ws(_inline(title_el)) if title_el is not None else ""
    if title:
        lines.append(f"# {title}")

    # First org name under <author>
    org_name = ""
    org = root.find(f".//{NS}author//{NS}representedOrganization/{NS}name")
    if org is not None:
        org_name = _collapse_ws(_text(org.text))
    if org_name:
        lines.append(f"**Manufacturer:** {org_name}")

    # Document type from top-level <code displayName>
    top_code = root.find(f"{NS}code")
    if top_code is not None:
        dn = top_code.get("displayName") or ""
        if dn:
            lines.append(f"**Document type:** {dn}")

    # Product info (first manufacturedProduct/manufacturedProduct)
    mp = root.find(f".//{NS}subject/{NS}manufacturedProduct/{NS}manufacturedProduct")
    if mp is not None:
        name_el = mp.find(f"{NS}name")
        if name_el is not None:
            nm = _collapse_ws(_inline(name_el))
            if nm:
                lines.append(f"**Product name:** {nm}")
        form_el = mp.find(f"{NS}formCode")
        if form_el is not None:
            f = form_el.get("displayName") or ""
            if f:
                lines.append(f"**Form:** {f}")
        gen_el = mp.find(f"{NS}asEntityWithGeneric/{NS}genericMedicine/{NS}name")
        if gen_el is not None:
            g = _collapse_ws(_text(gen_el.text))
            if g:
                lines.append(f"**Generic:** {g}")

        # Active ingredients
        ings = []
        for ing in mp.findall(f"{NS}ingredient"):
            cls = (ing.get("classCode") or "").upper()
            if cls and cls != "ACTIB" and cls != "ACTIM":
                continue
            sub = ing.find(f"{NS}ingredientSubstance/{NS}name")
            sub_name = _collapse_ws(_text(sub.text)) if sub is not None else ""
            qty = ing.find(f"{NS}quantity")
            qty_str = ""
            if qty is not None:
                num = qty.find(f"{NS}numerator")
                den = qty.find(f"{NS}denominator")
                if num is not None:
                    nv = num.get("value") or ""
                    nu = num.get("unit") or ""
                    qty_str = f"{nv} {nu}".strip()
                if den is not None:
                    dv = den.get("value") or ""
                    du = den.get("unit") or ""
                    den_str = f"{dv} {du}".strip()
                    if den_str and den_str not in ("1", "1 1"):
                        qty_str = f"{qty_str}/{den_str}".strip("/")
            if sub_name:
                ings.append(f"{sub_name} {qty_str}".strip())
        if ings:
            lines.append("**Active ingredients:** " + "; ".join(ings))

    return "\n".join(lines)


def xml_to_markdown(xml_text: str) -> str:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return f"<!-- xml parse error: {e} -->"

    media = _media_map(root)
    parts = []
    header = _render_header(root)
    if header:
        parts.append(header)

    # Body sections
    body_sections = root.findall(f".//{NS}component/{NS}structuredBody/{NS}component/{NS}section")
    for sec in body_sections:
        # skip the metadata-only "SPL ... DATA ELEMENTS SECTION" sections — header already
        # captured the structured product info from them. Keep them only if they
        # have a non-empty <text> block.
        code_el = sec.find(f"{NS}code")
        code = code_el.get("code") if code_el is not None else ""
        if code in ("48780-1",):  # SPL data elements section
            text_el = sec.find(f"{NS}text")
            body = _render_text_block(text_el) if text_el is not None else ""
            if not body.strip():
                continue
        rendered = _render_section(sec, depth=2)
        if rendered:
            parts.append(rendered)

    md = "\n\n".join(p for p in parts if p)
    md = _resolve_image_anchors(md, media)
    # squash 3+ blank lines
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md


def _convert_chunk(xmls):
    return [xml_to_markdown(x) for x in xmls]


def process_shard(in_path: str, out_path: str, pool: ProcessPoolExecutor):
    table = pq.read_table(in_path)
    xmls = table.column("xml").to_pylist()

    chunks = [xmls[i : i + CHUNK_ROWS] for i in range(0, len(xmls), CHUNK_ROWS)]
    results = list(pool.map(_convert_chunk, chunks))
    md = [m for chunk in results for m in chunk]
    assert len(md) == len(xmls)

    md_array = pa.array(md, type=pa.string())
    new_table = table.append_column("markdown", md_array)

    pq.write_table(
        new_table,
        out_path,
        compression=COMPRESSION,
        compression_level=COMPRESSION_LEVEL,
        row_group_size=ROW_GROUP_ROWS,
    )


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    shards = sorted(f for f in os.listdir(SRC_DIR) if f.startswith("part-") and f.endswith(".parquet"))
    print(f"[md] {len(shards)} shards in {SRC_DIR}", flush=True)

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for shard in tqdm(shards, unit="shard"):
            in_path = os.path.join(SRC_DIR, shard)
            out_path = os.path.join(DST_DIR, shard)
            if os.path.exists(out_path):
                continue
            t0 = time.time()
            process_shard(in_path, out_path, pool)
            print(
                f"[md]   {shard} -> {os.path.getsize(out_path) / 1e9:.2f} GB in {time.time() - t0:.1f}s",
                flush=True,
            )

    print(f"[done] markdown shards in {DST_DIR}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
