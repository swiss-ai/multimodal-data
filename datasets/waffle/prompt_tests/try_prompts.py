"""Generate training-grade WAFFLE captions via Qwen3.5-397B.

One consolidated prompt — no ablations. Designed for dataset creation, not dev
analysis. Produces a single long, exhaustive, vividly-observed caption per
image with mandatory full transcription of any enumerated list in the image.
"""
import asyncio
import base64
import io
import time
from pathlib import Path

import httpx
import polars as pl
from PIL import Image

ENDPOINT = "http://172.28.33.228:8080/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-397B-A17B-xyixuan"
PARQUET = "/capstor/store/cscs/swissai/infra01/vision-datasets/raw/cooldown/tau-vailab___WAFFLE/parquet/permissive/train-00000.parquet"
N_SAMPLES = 12
CONCURRENCY = 8
MAX_LONGEST_SIDE = 2048
MAX_TOKENS = 2400


def resize_image_bytes(raw: bytes, target: int = MAX_LONGEST_SIDE) -> bytes:
    im = Image.open(io.BytesIO(raw))
    w, h = im.size
    if max(w, h) <= target:
        return raw
    s = target / max(w, h)
    im = im.convert("RGB").resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


# ------------- Persona selection (voice calibration only, not content) -------

def persona_for(high_level_type: str) -> str:
    """Short persona label (for output metadata)."""
    t = (high_level_type or "").lower()
    if "historic" in t:
        return "architectural historian / preservation documentation"
    if "religious" in t:
        return "sacred architecture"
    if "public" in t or "commercial" in t:
        return "civic / commercial architecture"
    if "infrastructure" in t or "transportation" in t or "healthcare" in t:
        return "civil engineering / industrial"
    if "industrial" in t:
        return "industrial heritage"
    if "palaces" in t or "mansions" in t or "residential" in t:
        return "domestic architecture / decorative arts"
    if "institutional" in t:
        return "institutional buildings"
    if "educational" in t:
        return "educational buildings"
    if "castles" in t or "fortresses" in t:
        return "military architecture"
    return "general architecture"


def persona_vocabulary_hint(high_level_type: str) -> str:
    """Vocabulary + focal priority hint for the prompt — NOT a role or voice."""
    t = (high_level_type or "").lower()
    if "historic" in t:
        return ("DOMAIN: historic preservation / HABS-style documentation. "
                "Vocabulary to use when the image shows it: drawing conventions "
                "(plan, elevation, section, detail), survey metadata (sheet "
                "number, survey code, accession number, draftsman credit), "
                "construction era indicators, structural framing types. "
                "FOCAL PRIORITY: every label, annotation, dimension, and "
                "title-block field.")
    if "religious" in t:
        return ("DOMAIN: sacred / ecclesiastical architecture. "
                "Vocabulary: nave, apse, narthex, transept, chancel, belfry, "
                "bell tower, spire, clerestory, rose window, lancet, "
                "tympanum, pointed/round/horseshoe arch, vault, buttress, "
                "ambulatory, crypt, pulpit, pew, liturgical furnishings. "
                "FOCAL PRIORITY: vertical circulation, plan symmetry, "
                "fenestration type, louvered belfry openings.")
    if "public" in t or "commercial" in t:
        return ("DOMAIN: civic / commercial architecture. "
                "Vocabulary: colonnade, peristyle, pediment, entablature, "
                "coffer, portico, rotunda, lobby, concourse, vestibule, "
                "arcade, balustrade, dentil course, frieze. "
                "FOCAL PRIORITY: axial symmetry, ceremonial progression, "
                "ornament programs, scale indicators.")
    if "infrastructure" in t or "transportation" in t or "healthcare" in t or "industrial" in t:
        return ("DOMAIN: industrial / infrastructure. "
                "Vocabulary: structural framing (I-beam, truss, girder), "
                "mechanical systems (turbine, shaft, pulley, belt drive, "
                "conveyor, hopper, bin), power generation, flow/process "
                "diagrams, material flow arrows, foundations, sill plates. "
                "FOCAL PRIORITY: machinery layout, equipment keys, "
                "manufacturer marks, patent dates, dimension schedules.")
    if "palaces" in t or "mansions" in t or "residential" in t:
        return ("DOMAIN: domestic architecture / decorative arts. "
                "Vocabulary: moulding profiles (ogee, cavetto, cyma recta/"
                "reversa, torus, astragal, bead, fillet), cornice, chair "
                "rail, baseboard, panel, wainscot, dado, mantel, overmantel, "
                "door/window casing, jamb, architrave. "
                "FOCAL PRIORITY: joinery profiles, room-by-room finish "
                "documentation, stylistic period cues (Georgian, Federal, "
                "Adam, Greek Revival, etc.) when substantiated.")
    if "institutional" in t:
        return ("DOMAIN: institutional buildings (prisons, hospitals, "
                "schools, courthouses). "
                "Vocabulary: cell block, ward, corridor, ranges, surveillance "
                "fenestration, grille form, barred opening, control room, "
                "security partition, yard level, road level. "
                "FOCAL PRIORITY: fenestration typology, security hardware, "
                "compartmentalization, circulation hierarchy.")
    if "educational" in t:
        return ("DOMAIN: educational buildings. "
                "Vocabulary: classroom, auditorium, gymnasium, corridor, "
                "administration wing, stairwell, entrance lobby, court, "
                "wing arrangement. "
                "FOCAL PRIORITY: classroom layout, corridor circulation, "
                "wing symmetry, ceiling/flooring legend material codes.")
    if "castles" in t or "fortresses" in t:
        return ("DOMAIN: military architecture / fortification. "
                "Vocabulary: bastion, ravelin, curtain wall, parapet, "
                "merlon, crenellation, embrasure, loophole, barbican, "
                "moat, glacis, casemate, keep, gatehouse. "
                "FOCAL PRIORITY: defensive geometry, sightlines, "
                "masonry coursing, gun emplacements.")
    return ("DOMAIN: architecture (general). "
            "Use standard architectural vocabulary when specific features "
            "are visible. FOCAL PRIORITY: labels, dimensions, title block.")


# ------------- The prompt --------------------------------------------------

def build_system_prompt(vocabulary_hint: str) -> str:
    # vocabulary_hint ignored — reverted to no-persona V5 after vocab-hint
    # version regressed on scattered-label handling (Sample 0 Maloof).
    return """You produce captions for a vision-language training dataset. A downstream model will learn from these captions, so every word must be clean training signal: dense visual description, grounded in what is literally visible in the image. Your job is to REPORT what is on the page, not to INTERPRET, narrate, evaluate, or tell its story.

Write ONE description, as long as the image warrants. No headers, no bullets, no preamble ("This image shows...", "The drawing depicts...", "In the foreground..."). No closing summary ("Overall,", "In summary", "A testament to..."). Begin with a concrete description of what occupies the image and continue until every visible element has been described.

HARD RULES

1. DESCRIBE, DO NOT INTERPRET. Report what is drawn, labeled, shown, hatched, dimensioned. Do NOT explain function, history, purpose, intent, or meaning. BAD: "the machinery dictates the primary grinding axis", "a testament to industrial adaptation", "reads like a catalog of engineering", "sits ready for the final stage of processing". GOOD: "four circled numerals labeled 1 through 4 sit along a straight line running east-west, each labeled 'Flour Roller Mills'".

2. NO SPECULATION. Ban these words and their synonyms entirely: "likely", "perhaps", "possibly", "suggesting", "evoking", "reminiscent of", "characteristic of", "indicative of", "as if", "akin to", "hints at", "speaks to", "tells a story", "seems", "appears to". If you are tempted to hedge, you are interpreting — stop and report what you actually see instead.

3. NO EVALUATIVE OR LITERARY LANGUAGE. Ban these unless directly justified by specific visible evidence: "stark", "surgical", "meticulous", "exquisite", "grand", "monumental", "delicate", "intricate", "elegant", "striking", "handsome", "pure", "unified", "masterful", "testament to". "Hand-drawn" is allowed because ink on paper is visible; "crisp linework" is NOT unless you are distinguishing it from visibly blurred linework in the same image.

4. NO CONTENT NOT VISIBLE IN THE IMAGE. Do NOT reference anything outside the frame ("the unseen water wheel", "the adjacent room", "the hidden foundation"). Do NOT import dates, names, histories, or functional narratives that aren't printed on the page. If a date, name, or description appears in the image's text (title block, annotations, stamps), you may quote it exactly.

5. TRANSCRIBE ALL LEGIBLE TEXT. Labels, dimensions, scale indicators, sheet numbers, survey codes, project titles, draftsmen's names, dates, manufacturer marks, patent numbers, location text, key map notes, legend keys — transcribe them verbatim, in quotes. Building names printed in title blocks ARE visible text; include them.

6. TRANSCRIBE DISCRETE ENUMERATED LISTS IN FULL. This rule applies ONLY to bounded, clearly-delimited lists: a numbered legend/key, a window/door/profile schedule, a parts index, a multi-view callout set (PLAN-ON-A-A, PLAN-ON-B-B, ...), a bounded table with labeled rows. When such a list is present, transcribe every row. A 22-item legend yields 22 entries; a 35-row schedule yields 35 rows. Never write "among others" or "and so on" for these.

   This rule does NOT apply to scattered labels distributed across a drawing — tree survey dots, planting schedule points, grid reference numbers, contour elevation tags, PP-codes dispersed around a site plan, or any set of markers that would require listing dozens-to-hundreds of individual numbers. For these, DO NOT attempt exhaustive transcription. Instead describe them as a class with: (a) approximate count or density ("roughly 200 numbered circles", "dozens of elevation tags"), (b) the range or pattern of the numbering you can confidently read ("numbering runs from 1 to approximately 220", "elevations visible from 1535 down to 1480"), (c) 3-5 concrete example numbers you can literally read from the image. Do not generate sequences beyond what you can actually see. If in doubt, transcribe fewer examples rather than more.

7. NEVER CONFABULATE TO SATISFY A RULE. If you cannot clearly read an entry, omit it. If a list is too dense to transcribe fully, describe it as a class (see rule 6). It is better to be brief and honest than to produce a plausible-sounding but invented sequence. Duplicated entries, implausible sequences, or patterned extrapolations are hallucination — worse than no transcription.

8. ENUMERATE EACH VIEW SEPARATELY. If a sheet contains multiple plans, sections, elevations, or diagrams, describe each as a distinct block: name it ("First Floor Plan", "PLAN-ON-C-C", "Section A-A", "Roof Plan"), then describe what is drawn within it. Do not fold them into a generic composite.

9. MATERIAL INFERENCE IS ALLOWED BUT LIMITED. You may name a material when line weight, hatching, or stippling conventionally denotes it (stippled fill = masonry/stone; diagonal hatch = cut section; solid = wall; dashed = hidden/above). When you do, state the visual cue: "diagonal hatching denotes cut masonry", not just "masonry walls".

10. NEUTRAL VOICE. No "we see", no "the viewer", no "the eye", no first-person. No rhetorical questions. No narrative arc. Just description. Vary sentence openings to avoid starting every sentence with "The plan..." / "The drawing..." / "There is...", but do not introduce voice in the process.

11. PUNCTUATION AND QUOTES. Put every piece of quoted text in the image inside double quotes. Preserve exact capitalization and punctuation from the image.

Output begins with the first word of the description. No preface.
"""


def build_user_prompt() -> str:
    return "Caption this image following the rules in the system prompt."


# ------------- Inference ---------------------------------------------------

async def ask(client, sem, tag, img_b64, system_prompt):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": build_user_prompt()},
            ]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with sem:
        t0 = time.time()
        try:
            r = await client.post(ENDPOINT, json=body, timeout=2400.0)
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            reply = f"ERROR: {e}"
        dt = time.time() - t0
    words = len(reply.split())
    print(f"[{tag}] {dt:.1f}s  {words}w", flush=True)
    return tag, reply, dt


def pick_diverse_samples(df: pl.DataFrame, n: int) -> list[dict]:
    """Round-robin across high_level_building_type to maximize diversity."""
    from collections import defaultdict
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in df.iter_rows(named=True):
        by_cat[row.get("high_level_building_type") or "other"].append(row)
    out = []
    cats = list(by_cat.keys())
    i = 0
    while len(out) < n and any(by_cat[c] for c in cats):
        c = cats[i % len(cats)]
        if by_cat[c]:
            out.append(by_cat[c].pop(0))
        i += 1
    return out[:n]


async def main():
    df = pl.read_parquet(PARQUET).head(400)
    samples = pick_diverse_samples(df, N_SAMPLES)

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    for f in list(out_dir.glob("s*.txt")) + list(out_dir.glob("sample_*.jpg")):
        f.unlink()

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    meta = {}

    async with httpx.AsyncClient() as client:
        for i, row in enumerate(samples):
            resized = resize_image_bytes(row["image_bytes"])
            img_b64 = base64.b64encode(resized).decode()
            (out_dir / f"sample_{i}.jpg").write_bytes(resized)

            persona = persona_for(row.get("high_level_building_type", ""))
            vocab_hint = persona_vocabulary_hint(row.get("high_level_building_type", ""))
            sys_prompt = build_system_prompt(vocab_hint)
            tag = f"s{i:02d}"
            im = Image.open(io.BytesIO(resized))
            print(f"sample {i:2d}: {row['building_name']} — "
                  f"{row['high_level_building_type']} / {row['building_type']} "
                  f"— {im.size} {len(resized)//1024}KB", flush=True)
            meta[tag] = (i, row, persona, sys_prompt)
            tasks.append(ask(client, sem, tag, img_b64, sys_prompt))

        print(f"\nfiring {len(tasks)} captions with concurrency={CONCURRENCY}\n",
              flush=True)
        t0 = time.time()
        results = await asyncio.gather(*tasks)
        print(f"\nall done in {time.time()-t0:.1f}s\n", flush=True)

    for tag, reply, dt in sorted(results):
        i, row, persona, sys_prompt = meta[tag]
        (out_dir / f"{tag}.txt").write_text(
            f"PERSONA: {persona}\n\n"
            f"BUILDING (for reference, not fed to model): "
            f"{row['building_name']} | {row['building_type']} | "
            f"{row['high_level_building_type']} | {row['city']}, "
            f"{row['state']}, {row['country']}\n\n"
            f"CAPTION ({dt:.1f}s, {len(reply.split())} words):\n{reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
