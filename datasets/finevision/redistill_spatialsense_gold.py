"""SpatialSense redistill using ORIGINAL gold-verified bboxes + Qwen prose.

Source: princeton-vl SpatialSense Zenodo release
  https://zenodo.org/records/8104370   (CC-BY-2.0)

Pipeline:
  1. Load annotations.json — 11,569 images, 17,498 (subject, predicate, object)
     triples with human-VERIFIED bboxes and Yes/No labels.
  2. Filter to train split ONLY (user preference).
  3. For each annotation: load image, convert bbox from native [Y1,Y2,X1,X2]
     pixel coords to our standard [X1,Y1,X2,Y2] normalized 0-1000.
  4. Send to Qwen with the gold-everything context, asking for ONLY natural
     prose. Qwen does NOT decide Yes/No, does NOT generate bboxes — just writes
     a 1-2 sentence description that uses the gold bboxes inline.
  5. Output: spatialsense_recap_gold/ with one row per image (multi-turn from
     the multiple relations per image), in the same {images, texts, source}
     schema as our other FineVision processed parquets.

Fan-out: 32 concurrent requests round-robin across worker URLs from
/tmp/ready_workers.txt. Idempotent at shard level (10 shards by image-id mod).
"""

from __future__ import annotations
import base64
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import requests


ANN_PATH = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/spatialsense_original/annotations.json")
IMG_ROOT = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/hf_downloads/spatialsense_original/images")
OUT_DIR = Path("/capstor/store/cscs/swissai/infra01/vision-datasets/processed/sft/finevision/spatialsense_recap_gold")
WORKERS_FILE = Path("/tmp/ready_workers.txt")
MODEL = "Qwen/Qwen3.6-27B-xyixuan"
CONCURRENCY = 32
TIMEOUT = 60
RETRIES = 2
SAMPLE_EVERY = 100
SHARD_COUNT = 10  # split output into 10 shards by image-id hash


# Prompt: gold everything → Qwen writes natural prose, with the EXACT gold
# bboxes (we pre-format them and tell Qwen to copy-paste into the output).
SYSTEM_PROMPT = """You phrase a spatial relation in natural English for a training dataset.

You are GIVEN:
  - An image
  - A spatial relation question: "Is the SUBJECT PREDICATE the OBJECT?"
  - The GOLD answer: Yes or No (human-verified)
  - GOLD bounding boxes for SUBJECT and OBJECT (human-verified, format: [X_min, Y_min, X_max, Y_max] normalized 0-1000)

Your job: write ONE sentence in this exact format (no extra text before or after):

<answer>Yes</answer>. <description>The <object>SUBJECT_NAME</object><bbox>[X1,Y1,X2,Y2]</bbox> is RELATION the <object>OBJECT_NAME</object><bbox>[X1,Y1,X2,Y2]</bbox>, [optional brief 3D/scene context].</description>

OR

<answer>No</answer>. <description>The <object>SUBJECT_NAME</object><bbox>[X1,Y1,X2,Y2]</bbox> is RELATION the <object>OBJECT_NAME</object><bbox>[X1,Y1,X2,Y2]</bbox>, [optional brief 3D/scene context].</description>

Rules:
  1. Copy the GIVEN bboxes VERBATIM into the output — DO NOT re-detect or re-estimate them.
  2. <answer>: ALWAYS use the GIVEN gold Yes/No exactly. Do NOT decide it yourself.
  3. RELATION:
       - If gold = "Yes": RELATION = the asked predicate exactly (e.g., "on", "behind", "to the left of").
       - If gold = "No":  RELATION = a CORRECT alternative relation you can verify from the image (terms: above, below, to the left of, to the right of, behind, in front of, on top of, under, beside, next to, inside, outside, near). NEVER write "X is not <predicate> Y".
  4. Reason in 3D for "behind"/"in front of" — depth, occlusion, perspective. 2D bbox overlap alone is NOT enough.
  5. NEVER use: "elsewhere", "different relationship", "appears to", "seems to", "may be", "possibly", "perhaps".
  6. Keep the optional context short and grounded (1 brief clause max).

Example (gold = Yes, predicate = "on"):
INPUT: subject=cat <bbox>[110,460,272,868]</bbox>, predicate=on, object=ground <bbox>[10,804,989,998]</bbox>, gold=Yes
OUTPUT: <answer>Yes</answer>. <description>The <object>cat</object><bbox>[110,460,272,868]</bbox> is on the <object>ground</object><bbox>[10,804,989,998]</bbox>, lying on a brown tile floor.</description>

Example (gold = No, predicate = "under"):
INPUT: subject=cat <bbox>[110,460,272,868]</bbox>, predicate=under, object=mirror <bbox>[0,690,569,560]</bbox>, gold=No
OUTPUT: <answer>No</answer>. <description>The <object>cat</object><bbox>[110,460,272,868]</bbox> is to the right of the <object>mirror</object><bbox>[0,690,569,560]</bbox>, sitting beside it on the floor.</description>"""

RE_ANSWER = re.compile(r"<answer>\s*(Yes|No)\s*</answer>", re.IGNORECASE)
RE_DESC = re.compile(r"<description>(.*?)</description>", re.IGNORECASE | re.DOTALL)


def load_workers() -> list[str]:
    return [u.strip() for u in WORKERS_FILE.read_text().splitlines() if u.strip()]


def native_to_norm(bbox_native: list[int], width: int, height: int) -> list[int]:
    """Convert [Y_min, Y_max, X_min, X_max] pixels → [X1, Y1, X2, Y2] normalized 0-1000."""
    y1, y2, x1, x2 = bbox_native
    X1 = max(0, min(1000, int(round(x1 / max(1, width) * 1000))))
    Y1 = max(0, min(1000, int(round(y1 / max(1, height) * 1000))))
    X2 = max(0, min(1000, int(round(x2 / max(1, width) * 1000))))
    Y2 = max(0, min(1000, int(round(y2 / max(1, height) * 1000))))
    if X1 > X2: X1, X2 = X2, X1
    if Y1 > Y2: Y1, Y2 = Y2, Y1
    return [X1, Y1, X2, Y2]


def resolve_image_path(url: str) -> Path | None:
    """Map annotation URL to local image path. Flickr URLs → images/flickr/<basename>;
    NYU URLs (which start with /images/nyu/...) → images/nyu/<basename>."""
    if "/nyu/" in url or url.startswith("/images/nyu/"):
        base = url.split("/")[-1]
        p = IMG_ROOT / "nyu" / base
    else:
        base = url.split("/")[-1]
        p = IMG_ROOT / "flickr" / base
    return p if p.exists() else None


def to_data_url(image_path: Path) -> str:
    b = image_path.read_bytes()
    ext = image_path.suffix.lower().lstrip(".") or "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(b).decode()}"


def build_user_msg(question: str, subj_name: str, subj_bbox: list[int],
                   obj_name: str, obj_bbox: list[int], predicate: str, gold: str) -> str:
    sbox = f"[{subj_bbox[0]},{subj_bbox[1]},{subj_bbox[2]},{subj_bbox[3]}]"
    obox = f"[{obj_bbox[0]},{obj_bbox[1]},{obj_bbox[2]},{obj_bbox[3]}]"
    return (
        f"Question: {question}\n"
        f"subject = {subj_name} <bbox>{sbox}</bbox>\n"
        f"predicate = {predicate}\n"
        f"object = {obj_name} <bbox>{obox}</bbox>\n"
        f"gold = {gold}"
    )


def call_qwen(worker_url: str, image_data_url: str, user_text: str) -> str | None:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": user_text},
            ]},
        ],
        "max_tokens": 220,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(f"{worker_url}/v1/chat/completions", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            c = (msg.get("content") or msg.get("reasoning") or "").strip()
            if c:
                return c
        except Exception:
            if attempt == RETRIES:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def parse_qwen_output(text: str, expected_label: str) -> str | None:
    """Returns the validated integrated answer string, or None if invalid."""
    am = RE_ANSWER.search(text)
    if not am: return None
    if am.group(1).capitalize() != expected_label:
        # Qwen sometimes overrides the gold; force it back
        text = re.sub(r"<answer>\s*(Yes|No)\s*</answer>", f"<answer>{expected_label}</answer>", text, count=1, flags=re.IGNORECASE)
    dm = RE_DESC.search(text)
    if not dm: return None
    desc = dm.group(1).strip()
    if not desc: return None
    return f"{expected_label}. {desc}"


def process_image_entry(entry: dict, workers: list[str], wid_seed: int) -> dict | None:
    """For one image with N annotations, produce a row with N turns."""
    if entry.get("split") != "train":
        return None
    img_path = resolve_image_path(entry["url"])
    if img_path is None:
        return None
    width = entry["width"]
    height = entry["height"]

    try:
        img_bytes = img_path.read_bytes()
    except Exception:
        return None
    img_url = to_data_url(img_path)

    turns = []
    for j, ann in enumerate(entry.get("annotations", [])):
        subj = ann["subject"]
        obj = ann["object"]
        subj_box = native_to_norm(subj["bbox"], width, height)
        obj_box = native_to_norm(obj["bbox"], width, height)
        gold = "Yes" if ann["label"] else "No"

        question = f"Is the {subj['name']} {ann['predicate']} the {obj['name']}?"
        user_msg = build_user_msg(question, subj["name"], subj_box, obj["name"], obj_box,
                                  ann["predicate"], gold)
        worker = workers[(wid_seed + j) % len(workers)]
        out = call_qwen(worker, img_url, user_msg)
        if out is None:
            continue
        validated = parse_qwen_output(out, gold)
        if validated is None:
            continue

        # The user-facing question (FineVision-style chat format)
        user_question = f"<image>\n{question}"
        turns.append({"user": user_question, "assistant": validated})

    if not turns:
        return None

    # Build {images, texts, source} row matching FineVision schema
    img_struct = [{"bytes": img_bytes, "path": img_path.name}]
    return {"images": img_struct, "texts": turns, "source": "spatialsense_gold"}


def shard_index_for(entry: dict) -> int:
    """Deterministic shard assignment by URL hash → keeps shards balanced."""
    import hashlib
    h = hashlib.md5(entry["url"].encode()).hexdigest()
    return int(h[:8], 16) % SHARD_COUNT


def main():
    workers = load_workers()
    if not workers:
        print("no workers in /tmp/ready_workers.txt", file=sys.stderr)
        sys.exit(1)
    print(f"workers ({len(workers)}): {workers[:3]}...", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading annotations from {ANN_PATH}", flush=True)
    all_entries = json.load(ANN_PATH.open())
    train_entries = [e for e in all_entries if e.get("split") == "train"]
    print(f"  total: {len(all_entries):,}  train: {len(train_entries):,}", flush=True)
    total_relations = sum(len(e.get("annotations", [])) for e in train_entries)
    print(f"  total train relations: {total_relations:,}", flush=True)

    # Assign each entry to a shard, then process shard-by-shard for idempotency.
    by_shard: list[list[dict]] = [[] for _ in range(SHARD_COUNT)]
    for e in train_entries:
        by_shard[shard_index_for(e)].append(e)

    schema = pa.schema([
        ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())]))),
        ("texts", pa.list_(pa.struct([("user", pa.string()), ("assistant", pa.string())]))),
        ("source", pa.string()),
    ])

    t0 = time.time()
    total_kept = total_turns = 0
    for s_i, entries in enumerate(by_shard):
        out_path = OUT_DIR / f"train-{s_i:05d}-of-{SHARD_COUNT:05d}.parquet"
        if out_path.exists() and out_path.stat().st_size > 0:
            n = pq.read_metadata(out_path).num_rows
            total_kept += n
            print(f"  [shard {s_i}] SKIP (already done, {n:,} rows)", flush=True)
            continue
        t_shard = time.time()
        results: list[dict | None] = [None] * len(entries)
        base_seed = random.randint(0, 1_000_000)
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = {ex.submit(process_image_entry, e, workers, base_seed + i): i for i, e in enumerate(entries)}
            done = 0
            for f in as_completed(futs):
                i = futs[f]
                r = f.result()
                results[i] = r
                done += 1
                if done % SAMPLE_EVERY == 0:
                    nz = sum(1 for x in results[:done] if x is not None)
                    samples = [x for x in results[:done] if x is not None]
                    if samples:
                        s = random.choice(samples)
                        t0_turn = s["texts"][0]
                        sys.stderr.write(
                            f"  [shard {s_i} @ {done}/{len(entries)} kept={nz}] "
                            f"Q={t0_turn['user'][:100]!r} A={t0_turn['assistant'][:180]!r}\n")

        kept_rows = [r for r in results if r is not None]
        turn_count = sum(len(r["texts"]) for r in kept_rows)
        if kept_rows:
            tbl = pa.Table.from_pylist(kept_rows, schema=schema)
            tmp = out_path.with_suffix(".parquet.tmp")
            pq.write_table(tbl, str(tmp), compression="zstd")
            tmp.rename(out_path)
        dt = time.time() - t_shard
        total_kept += len(kept_rows)
        total_turns += turn_count
        print(f"  [shard {s_i}] images in={len(entries):>5,} kept={len(kept_rows):>5,} "
              f"turns={turn_count:>5,} in {dt:.0f}s ({len(entries)/dt:.1f} img/s)", flush=True)

    print(f"\n=== TOTAL ===")
    print(f"  images kept: {total_kept:,}")
    print(f"  turns:       {total_turns:,}")
    print(f"  elapsed:     {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
