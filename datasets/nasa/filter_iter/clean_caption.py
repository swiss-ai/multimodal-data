"""Strip boilerplate and metadata noise from NASA captions.

Three cleanup layers, applied in order:

1. TAIL CUT — from the earliest occurrence of any end-of-caption boilerplate
   marker (Photo credit, social-media links, image-use-policy, etc.) to the
   end of the string.

2. INLINE STRIP — short attribution parentheticals anywhere in the caption
   (P.I., Principal Investigator, JPL ref. No., trailing NASA/photographer
   attributions, Image Credit parens).

3. LEADING STRIP — redundant archive headers at the start (the nasa_id
   already carries this info): e.g. "AS11-37-5458 (20 July 1969) --- ".

Then strip residual HTML tags and collapse whitespace.
"""
import re

# Layer 1 — everything from the first match to end of string is dropped.
_TAIL_MARKERS = [
    r"\bPhoto\s+[Cc]redit\s*:",
    r"\bPHOTO\s+CREDIT\s*:",
    r"\bImage\s+[Cc]redit\s*:",
    r"\bNASA\s+image\s+use\s+policy",
    r"\bFor\s+more\s+information,?\s+visit\b",
    r"\bTo\s+learn\s+more\s+about\b[^.]*?\bvisit\b",
    r"\bFollow\s+us\s+on\b",
    r"\bLike\s+us\s+on\b",
    r"\bFind\s+us\s+on\b",
    r"<b>\s*NASA\s+Goddard\b",
    r"\bPhotograph\s+published\s+in\b",
    r"\bGoddard\s+plays\s+a\s+leading\s+role\b",
    r"<b>\s*<a\s+href",
]

# Layer 2 — inline parentheticals removed wherever they appear.
_INLINE_STRIPS = [
    # "(P.I. S Hipskind)", "(P. I.: Name)"
    r"\s*\(\s*P\.?\s*I\.?[\s.:]+[^)]{0,80}?\)",
    # "(Principal Investigator: Name)"
    r"\s*\(\s*Principal\s+Investigator[\s:]+[^)]{0,80}?\)",
    # "(JPL ref. No. P-21148)"
    r"\s*\(\s*JPL\s+ref\.?\s*No\.?[^)]{0,40}?\)",
    # Trailing "(NASA/Bill Ingalls)" style attributions — anywhere short paren
    r"\s*\(\s*NASA\s*/\s*[A-Za-z0-9][^)]{0,80}?\)",
]

# Layer 3 — only at start of string.
_LEADING_STRIPS = [
    # "AS11-37-5458 (20 July 1969) --- " / "STS059-S-001 (November 1993) --- "
    r"^[A-Za-z0-9][A-Za-z0-9_\-]{3,}\s*\([^)]{1,40}\)\s*-{2,}\s+",
]

_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
_WS_RE = re.compile(r"\s+")
_TAIL_RE = re.compile("|".join(_TAIL_MARKERS), re.IGNORECASE)
_INLINE_RE = re.compile("|".join(_INLINE_STRIPS), re.IGNORECASE)
_LEADING_RES = [re.compile(p, re.IGNORECASE) for p in _LEADING_STRIPS]


def clean_caption(text: str) -> str:
    if not text:
        return text
    # Layer 1 — tail cut.
    m = _TAIL_RE.search(text)
    if m:
        text = text[: m.start()]
    # Layer 2 — inline strips.
    text = _INLINE_RE.sub(" ", text)
    # Layer 3 — leading archive-header strip.
    for r in _LEADING_RES:
        text = r.sub("", text, count=1)
    # Strip any stray HTML tags left in the retained portion.
    text = _HTML_TAG_RE.sub(" ", text)
    # Collapse whitespace and trim trailing separators.
    text = _WS_RE.sub(" ", text).strip()
    text = text.rstrip(" -_;:")
    return text


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from labeled_samples import LABELS

    changed = 0
    total_before = total_after = 0
    for nid, _lbl, cap in LABELS:
        cleaned = clean_caption(cap)
        total_before += len(cap)
        total_after += len(cleaned)
        if cleaned != cap:
            changed += 1
            print(f"\n--- {nid}  ({len(cap)} -> {len(cleaned)} chars)")
            print(f"BEFORE tail: ...{cap[-300:]!r}")
            print(f"AFTER  tail: ...{cleaned[-300:]!r}")
    print(f"\n{changed}/{len(LABELS)} captions changed.")
    print(f"total chars: {total_before} -> {total_after} ({(total_before-total_after)/total_before*100:.1f}% reduction)")
