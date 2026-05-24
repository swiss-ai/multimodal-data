"""
This script processes the raw videos for each take, filters out blurry and
duplicate frames, and saves the kept frames along with metadata.
"""

import json
import os

import av
import cv2
import imagehash
import numpy as np
from PIL import Image

DATA_DIR = os.environ.get("EGOEXO4D_DIR", "")
OUT_DIR = os.environ.get("OUT_DIR", "/tmp/ego/sample_A2")
TAKES_PATH = os.path.join(DATA_DIR, "takes.json")
KEYSTEP_PATHS = [
    os.path.join(DATA_DIR, "annotations/keystep_train.json"),
    os.path.join(DATA_DIR, "annotations/keystep_val.json"),
]

LAPLACIAN_THRESHOLD = 400
HASH_DIFF_THRESHOLD = 4
MASK_RADIUS_RATIO = 0.8
FPS = 30
RESIZE_METHOD = cv2.INTER_LINEAR

H, W = 512, 512
CENTER = (W // 2, H // 2)
RADIUS = int(min(H, W) * MASK_RADIUS_RATIO / 2)
MASK = np.zeros((H, W), dtype=np.uint8)
cv2.circle(MASK, CENTER, RADIUS, 255, -1)  # type: ignore
MASK_BOOL = MASK == 255


def ttf(time_sec, fps=FPS) -> int:
    return int(round(time_sec * fps))


keystep_lookup = {}
for path in KEYSTEP_PATHS:
    with open(path, "r") as f:
        keystep_takes = json.load(f)
    for take in keystep_takes["annotations"].values():
        take_name = take["take_name"]
        segments = take["segments"]
        keystep_lookup[take_name] = []
        for segment in segments:
            keystep_lookup[take_name].append(
                {
                    "start_frame": ttf(segment["start_time"]),
                    "end_frame": ttf(segment["end_time"]),
                    "step_name": segment["step_name"],
                    "step_description": segment["step_description"],
                }
            )

with open(TAKES_PATH, "r") as f:
    takes = json.load(f)

for ti, take in enumerate(takes):

    root_dir = os.path.join(DATA_DIR, take["root_dir"])
    if (
        "frame_aligned_videos" not in take
        or "aria01" not in take["frame_aligned_videos"]
        or "rgb" not in take["frame_aligned_videos"]["aria01"]
        or "relative_path" not in take["frame_aligned_videos"]["aria01"]["rgb"]
    ):
        continue
    rel_path = take["frame_aligned_videos"]["aria01"]["rgb"]["relative_path"]
    path = os.path.join(root_dir, rel_path)

    name = take["take_name"]
    kept_dir = os.path.join(OUT_DIR, name, "frames/kept/rgb")
    blurry_dir = os.path.join(OUT_DIR, name, "frames/blurry")
    duplicate_dir = os.path.join(OUT_DIR, name, "frames/duplicate")
    os.makedirs(kept_dir, exist_ok=True)
    os.makedirs(blurry_dir, exist_ok=True)
    os.makedirs(duplicate_dir, exist_ok=True)

    print(f"Processing: {name}")

    container = av.open(path)
    container.streams.video[0].thread_type = "AUTO"
    total_frames = container.streams.video[0].frames

    kept_hashes = []
    counts = {"total": total_frames}

    metadata = []

    for fi, frame in enumerate(container.decode(video=0)):

        filename = f"frame_{fi:06d}.jpg"
        img = frame.to_ndarray(format="bgr24")

        # resize from 1408x1408 to 512x512
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # filter blurry images
        # compute variance from the center crop
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_values = laplacian[MASK_BOOL]
        score = laplacian_values.var()
        if score < LAPLACIAN_THRESHOLD:
            counts["blurry"] = counts.get("blurry", 0) + 1
            cv2.imwrite(os.path.join(blurry_dir, filename), img)
            metadata.append(
                {
                    "frame": fi,
                    "keep": False,
                    "reason": "blurry",
                    "laplacian_variance": score,
                }
            )
            continue

        # dedup
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cur_hash = imagehash.phash(img_pil)
        is_duplicate = False
        for existing_hash, _fi in kept_hashes:
            if (cur_hash - existing_hash) < HASH_DIFF_THRESHOLD:
                is_duplicate = True
                metadata.append(
                    {
                        "frame": fi,
                        "keep": False,
                        "reason": "duplicate",
                        "duplicate_of_frame": _fi,
                        "hamming_distance": cur_hash - existing_hash,
                    }
                )
                break
        if is_duplicate:
            counts["duplicate"] = counts.get("duplicate", 0) + 1
            cv2.imwrite(os.path.join(duplicate_dir, filename), img)
            continue

        kept_hashes.append((cur_hash, fi))
        counts["kept"] = counts.get("kept", 0) + 1
        cv2.imwrite(os.path.join(kept_dir, filename), img)
        metadata.append(
            {
                "frame": fi,
                "keep": True,
            }
        )

        print(f"Processed frame {fi + 1}/{total_frames}", end="\r")

    for mod in ["slam-left", "slam-right"]:
        rel_path = take["frame_aligned_videos"]["aria01"][mod]["relative_path"]
        path = os.path.join(root_dir, rel_path)
        kept_dir = os.path.join(OUT_DIR, name, "frames/kept", mod)
        os.makedirs(kept_dir, exist_ok=True)
        container = av.open(path)
        container.streams.video[0].thread_type = "AUTO"
        for fi, frame in enumerate(container.decode(video=0)):
            if any(fi == _fi for _, _fi in kept_hashes):
                filename = f"frame_{fi:06d}.jpg"
                img = frame.to_ndarray(format="bgr24")
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(kept_dir, filename), img)

    print(f"Take summary: {counts}")
    take["__summary"] = counts
    take["__frames"] = "@@metadata@@"
    take["__keysteps"] = "@@keysteps@@"
    final = json.dumps(take, indent=2)

    metadata_formatted = "[\n    " + ",\n    ".join(json.dumps(x) for x in metadata) + "\n  ]"

    keysteps = keystep_lookup.get(name, [])
    keysteps_formatted = "[\n    " + ",\n    ".join(json.dumps(x) for x in keysteps) + "\n  ]"

    final = final.replace('"@@metadata@@"', metadata_formatted)
    final = final.replace('"@@keysteps@@"', keysteps_formatted)

    with open(os.path.join(OUT_DIR, name, "metadata.json"), "w") as f:
        f.write(final)
