"""
Cleaning prompts for each Smithsonian WDS subset.

PROMPT_BY_SUBSET: dict mapping subset path (e.g. "tier1/nmaahc") to a prompt
template string with a single {caption} placeholder. The "default" key is
used for all subsets not listed explicitly.
"""

DEFAULT_PROMPT = """\
You are a museum caption editor. Clean and lightly polish the following raw Smithsonian museum record.

Rules:
1. Remove ALL physical dimension/measurement blocks wherever they appear:
   "H x W: …", "H x W x D: …", "overall: X cm", "sheet: X in.",
   "Image/Sheet: …", "Mount: …", "| X x Y x Z cm (…in.)", etc.
2. Remove ALL catalogue metadata: object-type labels ("Painting.", "Photograph."),
   creator/attribution lines ("Created by X, born Y-Z.", "Maker: X."),
   date labels ("Dated 1991."), medium labels ("Medium: cotton, silk…"),
   provenance ("From National Portrait Gallery.", "Place of origin: …"),
   pipe-separated metadata blocks, display-status notes ("Currently not on view"),
   and condition/status codes ("unused", "mint", "Mint X-cent; POD stamped on back.",
   "overall material", "overall color").
3. Remove technical spec dumps: semicolon-separated codes, abbreviations, or
   measurements that are not readable prose — e.g. "ra 0-350; 4 mounting holes;
   navy; j2m3, h8k2" or "55mm dia; range 0-130 deg c.; 2 position switch."
4. Keep ALL narrative and descriptive prose. You may lightly fix:
   - OCR line-break artifacts ("photog- raphy" → "photography")
   - Redundant title repetition (if the first sentence of the prose restates
     the title almost verbatim, drop the restatement)
   - Obvious grammatical glitches left by stripping
   Do NOT summarise, paraphrase the meaning, or add any new information.
5. Format: strip trailing punctuation from the title line if it is a standalone
   heading. Insert a newline between the title and descriptive prose.
   Do not add markdown, bullets, or headers.
6. If the remaining text consists only of a title, a single metadata word,
   or a meaningless fragment with no descriptive content, output the object
   title (the first phrase before any period or pipe) only.
7. Output ONLY the cleaned text. No preamble, no explanation.

Caption:
{caption}

Cleaned:"""

# tier1/nmafa captions begin with the material (e.g. "Copper alloy H x W x D: …")
# rather than a title. Strip the material line + dimensions, keep the description.
NMAFA_PROMPT = """\
You are a museum caption editor. Clean the following raw caption from the Smithsonian
National Museum of African Art. These records start with the medium/material, followed
by dimension blocks, followed by a descriptive sentence.

Rules:
1. Remove the leading material/medium token (e.g. "Copper alloy", "Bone, hair, plant fiber",
   "Distemper and gesso on wood") — it is metadata, not prose.
2. Remove ALL dimension/measurement blocks: "H x W: …", "H x W x D: …",
   "X × Y cm (…in.)", etc.
3. Keep ALL remaining descriptive text VERBATIM. Do not summarise, paraphrase,
   or add any new information.
4. Do not add markdown, bullets, or headers.
5. If remaining text would be empty or trivial, output the material token only.
6. Output ONLY the cleaned text. No preamble, no explanation.

Caption:
{caption}

Cleaned:"""

# tier2/other/npm captions contain philatelic condition notes and catalog codes
# mixed into the prose after stripping metadata. These need more specific removal.
NPM_PROMPT = """\
You are a museum caption editor. Clean and lightly polish the following raw caption
from the Smithsonian National Postal Museum.

Rules:
1. Remove ALL physical dimension/measurement blocks: "H x W: …", "Height x Width: …",
   "X x Y cm (…in.)", etc.
2. Remove ALL catalogue metadata: object-type labels, creator/attribution lines,
   date labels, medium labels, provenance ("From National Postal Museum.",
   "Place of origin: …"), pipe-separated metadata blocks.
3. Remove philatelic condition and catalog codes that appear at the start of the
   description or as standalone phrases:
   - Single condition words: "unused", "used", "mint"
   - Catalog condition lines: "Mint X-cent Y; 'POD' [Post Office Department] stamped
     on back.", "Issued imperforate.", "Perforated X."
   Keep technical production descriptions that are part of a sentence of prose
   (e.g. "line engraved on steel by Frederick Halpin, printed by Archer & Daly").
4. Keep ALL historical narrative and descriptive prose VERBATIM.
   You may lightly fix obvious grammatical glitches left by stripping.
5. Format: strip trailing punctuation from the title if it is a standalone heading.
   Insert a newline between title and prose. Do not add markdown, bullets, or headers.
6. If the remaining text has no descriptive prose, output the object title only.
7. Output ONLY the cleaned text. No preamble, no explanation.

Caption:
{caption}

Cleaned:"""

# tier2/other/sia captions are archive photo records. They contain USNM/SPS
# accession numbers, pipe-separated creator credits, medium/format lines,
# and archive location strings (box/folder/image-no/address) mixed with
# useful scene descriptions.
SIA_PROMPT = """\
You are a museum caption editor. Clean the following raw caption from the
Smithsonian Institution Archives. These records mix archive accession numbers,
creator credits, medium lines, and archive shelf-location strings with useful
scene descriptions.

Rules:
1. Remove accession/catalog numbers at the start of the description:
   "USNM No. XXXX", "SPS No. XXXX", "SPS Copy No. XXXX".
2. Remove ALL creator/attribution lines:
   "Created by [name] | [name] | ..., [year]."
3. Remove ALL medium/format metadata:
   "Glass negatives.", "Photographs.", "Cyanotypes (photographic prints); 8 x 10;",
   "Gelatin silver prints; 8 x 10;", "Stereoscopic photographs; 5 x 8;", etc.
4. Remove place-of-origin lines: "Place of origin: X."
5. Remove ALL archive shelf-location and reference strings, including:
   "Smithsonian Institution Archives, [Acc.|Record Unit] X, Box X, Folder X, Image No. SIA_XXXXXX"
   "Smithsonian Institution Archives Capital Gallery, Suite 3000, MRC 507; 600 Maryland Avenue, SW; Washington, DC 20024-2520"
   "Image No. SIA_XXXXXX" and any similar identifier or address block.
6. Keep ALL descriptive scene content VERBATIM: who or what is depicted,
   what is happening, architectural details, visible text in the image, etc.
7. The title (first phrase before the first period) is a location or event name
   — keep it as a heading line.
8. If remaining text would be empty or trivial, output the title only.
9. Output ONLY the cleaned text. No preamble, no explanation.

Caption:
{caption}

Cleaned:"""

PROMPT_BY_SUBSET: dict[str, str] = {
    "default": DEFAULT_PROMPT,
    "tier1/nmafa": NMAFA_PROMPT,
    "tier2/other/npm": NPM_PROMPT,
    "tier2/other/sia": SIA_PROMPT,
}
