#!/usr/bin/env python3
"""caption.py — Filter & caption pipeline for swisstopo map tiles.

Modes:
  filter-test     Pick 50 first-images from random shards, run filter only,
                  group results into passed/ and skipped/ directories.
  filter          Run filter on one shard, write per-image verdicts JSONL.
  filter-slurm    SLURM array worker — distributes shards for filter pass.
  caption-test    Sample N passed images across personas, caption, dump to dir.
  caption         Caption all passed images for one shard, write JSONL.
  caption-slurm   SLURM array worker — distributes shards for caption pass.
"""

import argparse
import base64
import json
import random
import tarfile
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SWISSTOPO_DATA_ROOT", ""))
OUTPUT_ROOT = Path(os.environ.get("SWISSTOPO_CAPTION_OUTPUT", "/tmp/toolbox/swisstopo_maps/outputs"))
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "154"))

GEMMA_PATH = os.environ.get("GEMMA_MODEL_PATH", "")

# Prompts
FILTER_PROMPT = """\
Look at this image. It is a tile from a Swiss topographic, cadastral, or land-survey map.

We want to keep tiles that contain enough SPECIFIC, STRUCTURED content for a \
detailed caption to teach a model something concrete. The bar is HIGH: a generic \
"rural area with some buildings and trees" caption is not useful.

This image is WORTH CAPTIONING if MOST of these are true:
- It shows a coherent settlement, neighbourhood, town centre, or industrial / urban \
area — not just a few scattered buildings in a forest or field.
- A meaningful road or street network is visible (multiple connected roads, junctions, \
named streets).
- Multiple readable labels are present (several place names, parcel numbers, road \
names, or building IDs — not just one or two place names).
- For cadastral / vermessung tiles: a clear cluster of parcels and building footprints \
with parcel numbers visible.
- For topographic tiles: built-up structure (a hamlet, village, or town), or a \
distinctive named feature (river crossing, railway station, lake shore with infrastructure).

This image should be SKIPPED if ANY of these are true:
- Mostly empty terrain (forest, meadow, alpine slope, fields) with only 1-3 isolated \
buildings, even if it has contour lines and one or two place names.
- Mostly white / off-map / outside Switzerland / lake centre — large blank areas.
- Map content is heavily cut off; only a fragment of useful content is visible at the edge.
- Text and symbols are too blurry, pixelated, or small to read.
- Generic alpine landscape: contours and forest with no real settlement.
- Corrupt or degenerate render.

When in doubt, SKIP. We have plenty of tiles; we want only the rich ones.

Respond with EXACTLY one of:
CAPTION — if this tile clearly has rich, specific content worth describing in detail
SKIP — otherwise

Your response (CAPTION or SKIP):"""


CAPTION_PROMPT = """\
Describe this place in detail. Use only what is visible on the map. Write a long, thorough, precise caption — describe the layout and how things connect, not just the names of features. Write in English. Do not begin with "This" or "A map". Avoid describing how the map is rendered or commenting on its scale, style, or purpose.
"""


def to_data_url(png_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('ascii')}"


def build_filter_messages(png_bytes: bytes) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_data_url(png_bytes)}},
                {"type": "text", "text": FILTER_PROMPT},
            ],
        }
    ]


def build_caption_messages(png_bytes: bytes) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": to_data_url(png_bytes)}},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]


def iter_first_per_shard(shard_indices: list[int]) -> list[dict]:
    """Yield the first .png + .json pair from each shard."""
    out = []
    for s in shard_indices:
        tar_path = DATA_ROOT / f"{s:05d}.tar"
        with tarfile.open(tar_path, "r") as tf:
            png_member = None
            json_member = None
            for m in tf:
                if not m.isfile():
                    continue
                if m.name.endswith(".png") and png_member is None:
                    png_member = m
                elif m.name.endswith(".json") and json_member is None:
                    json_member = m
                if png_member is not None and json_member is not None:
                    if png_member.name.split(".")[0] == json_member.name.split(".")[0]:
                        break
            assert png_member is not None and json_member is not None
            png_f = tf.extractfile(png_member)
            json_f = tf.extractfile(json_member)
            assert png_f is not None and json_f is not None
            png = png_f.read()
            meta = json.loads(json_f.read())
            out.append(
                {
                    "shard": s,
                    "key": png_member.name.rsplit(".", 1)[0],
                    "png": png,
                    "meta": meta,
                }
            )
    return out


def run_filter_test(model_path: str, n: int, seed: int):
    """Pick n random shards, take first sample from each, run filter, group."""
    from vllm import LLM, SamplingParams

    rng = random.Random(seed)
    shard_idx = rng.sample(range(NUM_SHARDS), n)
    print(f"Loading first sample from {n} random shards...")
    samples = iter_first_per_shard(shard_idx)

    import shutil

    out_dir = OUTPUT_ROOT / "filter_test_50"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pass_dir = out_dir / "passed"
    skip_dir = out_dir / "skipped"
    pass_dir.mkdir(parents=True, exist_ok=True)
    skip_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_path} ...")
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.80,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )

    params = SamplingParams(temperature=0.0, max_tokens=16)
    convos = [build_filter_messages(s["png"]) for s in samples]
    outputs = llm.chat(convos, sampling_params=params, use_tqdm=True)

    rows = []
    for s, o in zip(samples, outputs):
        verdict = o.outputs[0].text.strip().upper()
        passed = "CAPTION" in verdict
        target = pass_dir if passed else skip_dir
        name = (
            f"shard{s['shard']:05d}_{s['meta']['sample_id']}_"
            f"{s['meta']['layer'].split('.')[-1]}_s{s['meta']['scale']}.png"
        )
        (target / name).write_bytes(s["png"])
        rows.append(
            {
                "shard": s["shard"],
                "sample_id": s["meta"]["sample_id"],
                "layer": s["meta"]["layer"],
                "scale": s["meta"]["scale"],
                "img_w": s["meta"]["img_w"],
                "lang": s["meta"]["lang"],
                "verdict": "CAPTION" if passed else "SKIP",
                "raw": o.outputs[0].text.strip(),
                "filename": name,
            }
        )

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    n_pass = sum(r["verdict"] == "CAPTION" for r in rows)
    print(f"\n=== Filter test: {n_pass} passed, {n - n_pass} skipped ===")
    print(f"  passed → {pass_dir}")
    print(f"  skipped → {skip_dir}")
    print(f"  results.json → {out_dir / 'results.json'}")


def iter_shard_samples(shard_idx: int, chunk_size: int = 1000):
    """Stream every (.png, .json) pair from a shard in chunks."""
    tar_path = DATA_ROOT / f"{shard_idx:05d}.tar"
    pending: dict[str, dict] = {}
    chunk: list[dict] = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, _, ext = m.name.rpartition(".")
            if ext not in ("png", "json"):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            entry = pending.setdefault(stem, {"key": stem, "shard": shard_idx})
            if ext == "png":
                entry["png"] = data
            else:
                try:
                    entry["meta"] = json.loads(data)
                except Exception:
                    entry["meta"] = {}
            if "png" in entry and "meta" in entry:
                chunk.append(entry)
                pending.pop(stem)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def run_filter_shard(shard_idx: int, model_path: str, batch_size: int, out_dir: Path, llm=None):
    from vllm import LLM, SamplingParams

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
    if out_path.exists():
        print(f"  shard {shard_idx:05d}: already done ({out_path.name}), skipping")
        return llm

    if llm is None:
        print(f"Loading {model_path} ...")
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.80,
            dtype="bfloat16",
            max_model_len=8192,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 1},
        )

    params = SamplingParams(temperature=0.0, max_tokens=16)

    n_total = n_pass = 0
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as fo:
        for chunk in iter_shard_samples(shard_idx, chunk_size=1000):
            for batch_start in range(0, len(chunk), batch_size):
                batch = chunk[batch_start : batch_start + batch_size]
                convos = [build_filter_messages(s["png"]) for s in batch]
                outputs = llm.chat(convos, sampling_params=params, use_tqdm=False)
                for s, o in zip(batch, outputs):
                    raw = o.outputs[0].text.strip().upper()
                    verdict = "CAPTION" if "CAPTION" in raw else "SKIP"
                    n_total += 1
                    if verdict == "CAPTION":
                        n_pass += 1
                    rec = {
                        "shard": shard_idx,
                        "key": s["key"],
                        "sample_id": s["meta"].get("sample_id"),
                        "layer": s["meta"].get("layer"),
                        "scale": s["meta"].get("scale"),
                        "img_w": s["meta"].get("img_w"),
                        "lang": s["meta"].get("lang"),
                        "building_frac": s["meta"].get("building_frac"),
                        "verdict": verdict,
                    }
                    fo.write(json.dumps(rec) + "\n")
        fo.flush()
    tmp_path.rename(out_path)
    print(
        f"  shard {shard_idx:05d}: {n_pass}/{n_total} CAPTION ({n_pass / max(n_total, 1) * 100:.1f}%) → {out_path.name}"
    )
    return llm


def load_passed_keys(shard_idx: int, filter_dir: Path) -> dict[str, dict]:
    """key -> filter record, for verdicts == CAPTION only."""
    fp = filter_dir / f"shard_{shard_idx:05d}.jsonl"
    out = {}
    with open(fp) as f:
        for line in f:
            r = json.loads(line)
            if r["verdict"] == "CAPTION":
                out[r["key"]] = r
    return out


def iter_passed_samples(shard_idx: int, passed_keys: dict[str, dict], chunk_size: int = 500):
    """Stream (.png, .json) pairs from a shard, only for keys in passed_keys."""
    tar_path = DATA_ROOT / f"{shard_idx:05d}.tar"
    pending: dict[str, dict] = {}
    chunk: list[dict] = []
    with tarfile.open(tar_path, "r") as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, _, ext = m.name.rpartition(".")
            if ext not in ("png", "json"):
                continue
            if stem not in passed_keys:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            entry = pending.setdefault(stem, {"key": stem, "shard": shard_idx})
            if ext == "png":
                entry["png"] = data
            else:
                try:
                    entry["meta"] = json.loads(data)
                except Exception:
                    entry["meta"] = {}
            if "png" in entry and "meta" in entry:
                chunk.append(entry)
                pending.pop(stem)
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


def make_llm(model_path: str):
    from vllm import LLM

    return LLM(
        model=model_path,
        tokenizer=model_path,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.80,
        dtype="bfloat16",
        max_model_len=8192,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1},
    )


def caption_sampling_params():
    from vllm import SamplingParams

    return SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1024)


def run_caption_test(model_path: str, n: int, seed: int, filter_dir: Path, out_dir: Path):
    """Sample n passed images from one shard and caption them.

    Single shard keeps the tar walk to one (~100s on cold Lustre) so the
    test fits comfortably in a short slurm allocation.
    """
    import shutil
    import time

    t0 = time.time()
    rng = random.Random(seed)

    shard_files = sorted(filter_dir.glob("shard_*.jsonl"))
    assert shard_files, f"no filter outputs in {filter_dir}"
    fp = rng.choice(shard_files)
    shard_idx = int(fp.stem.split("_")[1])
    passed = [json.loads(l) for l in open(fp) if json.loads(l)["verdict"] == "CAPTION"]
    rng.shuffle(passed)
    chosen = passed[:n]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{time.time() - t0:6.1f}s] Loading {model_path} ...", flush=True)
    llm = make_llm(model_path)
    print(
        f"[{time.time() - t0:6.1f}s] Model ready, walking tar shard {shard_idx:05d}",
        flush=True,
    )

    keys = {r["key"]: r for r in chosen}
    samples: list[dict] = []
    for chunk in iter_passed_samples(shard_idx, keys, chunk_size=len(keys)):
        samples.extend(chunk)
    assert len(samples) == len(chosen), (len(samples), len(chosen))
    print(f"[{time.time() - t0:6.1f}s] Got {len(samples)} samples, captioning", flush=True)

    params = caption_sampling_params()
    convos = [build_caption_messages(s["png"]) for s in samples]
    outputs = llm.chat(convos, sampling_params=params, use_tqdm=True)

    rows = []
    for s, o in zip(samples, outputs):
        caption = o.outputs[0].text.strip()
        layer_tag = s["meta"]["layer"].split(".")[-1]
        base = f"shard{s['shard']:05d}_{s['meta']['sample_id']}_{layer_tag}_s{s['meta']['scale']}"
        (out_dir / f"{base}.png").write_bytes(s["png"])
        (out_dir / f"{base}.txt").write_text(caption + "\n")
        rows.append(
            {
                "shard": s["shard"],
                "key": s["key"],
                "sample_id": s["meta"].get("sample_id"),
                "layer": s["meta"].get("layer"),
                "scale": s["meta"].get("scale"),
                "img_w": s["meta"].get("img_w"),
                "lang": s["meta"].get("lang"),
                "caption": caption,
                "filename": f"{base}.png",
            }
        )
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    print(f"\n=== Caption test: {len(rows)} captions → {out_dir} ===")


def run_caption_shard(
    shard_idx: int,
    model_path: str,
    batch_size: int,
    filter_dir: Path,
    out_dir: Path,
    llm=None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{shard_idx:05d}.jsonl"
    if out_path.exists():
        print(f"  shard {shard_idx:05d}: already done, skipping")
        return llm

    tmp_path = out_path.with_suffix(".jsonl.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    passed = load_passed_keys(shard_idx, filter_dir)
    if not passed:
        out_path.write_text("")
        print(f"  shard {shard_idx:05d}: 0 passed, empty output")
        return llm

    if llm is None:
        print(f"Loading {model_path} ...")
        llm = make_llm(model_path)
    params = caption_sampling_params()

    n_total = 0
    with open(tmp_path, "w") as fo:
        for chunk in iter_passed_samples(shard_idx, passed, chunk_size=500):
            for batch_start in range(0, len(chunk), batch_size):
                batch = chunk[batch_start : batch_start + batch_size]
                convos = [build_caption_messages(s["png"]) for s in batch]
                outputs = llm.chat(convos, sampling_params=params, use_tqdm=False)
                for s, o in zip(batch, outputs):
                    caption = o.outputs[0].text.strip()
                    rec = {
                        "shard": shard_idx,
                        "key": s["key"],
                        "sample_id": s["meta"].get("sample_id"),
                        "layer": s["meta"].get("layer"),
                        "scale": s["meta"].get("scale"),
                        "img_w": s["meta"].get("img_w"),
                        "lang": s["meta"].get("lang"),
                        "building_frac": s["meta"].get("building_frac"),
                        "caption": caption,
                    }
                    fo.write(json.dumps(rec) + "\n")
                    n_total += 1
        fo.flush()
    tmp_path.rename(out_path)
    print(f"  shard {shard_idx:05d}: {n_total} captions → {out_path.name}")
    return llm


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "filter-test",
            "filter",
            "filter-slurm",
            "caption-test",
            "caption",
            "caption-slurm",
        ],
    )
    p.add_argument(
        "--n",
        type=int,
        default=50,
        help="filter-test / caption-test: number of samples",
    )
    p.add_argument("--shard", type=int, default=0, help="filter / caption: shard index to process")
    p.add_argument("--out", default="outputs/filter_full", help="output dir for per-shard JSONL")
    p.add_argument(
        "--filter-dir",
        default="outputs/filter_full",
        help="caption modes: directory of filter shard JSONLs",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--model", default=GEMMA_PATH)
    args = p.parse_args()

    def _resolve(path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else Path(__file__).parent / p

    if args.mode == "filter-test":
        run_filter_test(args.model, args.n, args.seed)
        return

    if args.mode == "filter":
        run_filter_shard(args.shard, args.model, args.batch_size, _resolve(args.out))
        return

    if args.mode == "filter-slurm":
        import os as _os

        worker_id = int(_os.environ["SLURM_ARRAY_TASK_ID"])
        num_workers = int(_os.environ.get("SLURM_ARRAY_TASK_COUNT", "32"))
        my_shards = [s for s in range(NUM_SHARDS) if s % num_workers == worker_id]
        print(f"Worker {worker_id}/{num_workers}: shards {my_shards}")
        out_dir = _resolve(args.out)
        llm = None
        for s in my_shards:
            llm = run_filter_shard(s, args.model, args.batch_size, out_dir, llm)
        return

    if args.mode == "caption-test":
        out_dir = _resolve(args.out if args.out != "outputs/filter_full" else "outputs/caption_test")
        run_caption_test(args.model, args.n, args.seed, _resolve(args.filter_dir), out_dir)
        return

    if args.mode == "caption":
        out_dir = _resolve(args.out if args.out != "outputs/filter_full" else "outputs/captions_full")
        run_caption_shard(args.shard, args.model, args.batch_size, _resolve(args.filter_dir), out_dir)
        return

    if args.mode == "caption-slurm":
        import os as _os

        worker_id = int(_os.environ["SLURM_ARRAY_TASK_ID"])
        num_workers = int(_os.environ.get("SLURM_ARRAY_TASK_COUNT", "32"))
        my_shards = [s for s in range(NUM_SHARDS) if s % num_workers == worker_id]
        print(f"Worker {worker_id}/{num_workers}: shards {my_shards}")
        out_dir = _resolve(args.out if args.out != "outputs/filter_full" else "outputs/captions_full")
        filter_dir = _resolve(args.filter_dir)
        llm = None
        for s in my_shards:
            llm = run_caption_shard(s, args.model, args.batch_size, filter_dir, out_dir, llm)
        return

    raise NotImplementedError(f"mode {args.mode}")


if __name__ == "__main__":
    main()
