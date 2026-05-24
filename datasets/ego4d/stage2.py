"""
This script processes the Ego4D dataset to extract key frames from video segments based on keystep annotations. It performs the following steps:
1. Loads keystep annotations and creates a lookup for quick access.
2. For each take, it identifies the relevant video segments and extracts frames.
3. It applies a blur check using the Laplacian variance and a duplicate check based on the distance to the middle frame of the segment.
4. The best candidate frame is saved, and its metadata is recorded.
5. Additionally, it extracts corresponding frames from SLAM videos for the selected key frames.
6. Finally, it compiles all the metadata into a JSON file for future reference
"""

import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import av
import cv2
import numpy as np

DATA_DIR = "/path/to/data/vision-datasets/egoexo4D"
OUT_DIR = "/tmp/ego/sample_mid2"
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
NUM_WORKERS = 64

# mask for selective laplacian
H, W = 512, 512
CENTER = (W // 2, H // 2)
RADIUS = int(min(H, W) * MASK_RADIUS_RATIO / 2)
MASK = np.zeros((H, W), dtype=np.uint8)
cv2.circle(MASK, CENTER, RADIUS, 255, -1)  # type: ignore
MASK_BOOL = MASK == 255

os.makedirs(OUT_DIR, exist_ok=True)


def ttf(time_sec, fps=FPS) -> int:
    return int(round(time_sec * fps))


def load_keystep_lookup():
    lookup = {}
    for path in KEYSTEP_PATHS:
        with open(path, "r") as f:
            data = json.load(f)
        annotations = data["annotations"]
        for take in annotations.values():
            take_name = take["take_name"]
            segments = []
            for segment in take["segments"]:
                segments.append(
                    {
                        "start_frame": ttf(segment["start_time"]),
                        "end_frame": ttf(segment["end_time"]),
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "step_name": segment["step_name"],
                        "step_description": segment["step_description"],
                    }
                )
            segments.sort(key=lambda x: x["start_frame"])
            lookup[take_name] = segments
    return lookup


if not os.path.exists("keystep_lookup.pkl"):
    GLOBAL_KEYSTEP_LOOKUP = load_keystep_lookup()
    with open("keystep_lookup.pkl", "wb") as f:
        pickle.dump(GLOBAL_KEYSTEP_LOOKUP, f)
else:
    with open("keystep_lookup.pkl", "rb") as f:
        GLOBAL_KEYSTEP_LOOKUP = pickle.load(f)


def process_single_take(arg):
    i, take_data = arg
    take_name = take_data["take_name"]

    segments = GLOBAL_KEYSTEP_LOOKUP.get(take_name)
    if not segments:
        return None

    mod = "rgb"
    root_dir = os.path.join(DATA_DIR, take_data["root_dir"])
    if (
        "frame_aligned_videos" not in take_data
        or "aria01" not in take_data["frame_aligned_videos"]
        or mod not in take_data["frame_aligned_videos"]["aria01"]
        or "relative_path" not in take_data["frame_aligned_videos"]["aria01"][mod]
    ):
        return None
    rel_path = take_data["frame_aligned_videos"]["aria01"][mod]["relative_path"]
    video_path = os.path.join(root_dir, rel_path)

    container = av.open(video_path)
    # container.streams.video[0].thread_type = "AUTO"

    stream = container.streams.video[0]
    time_base = stream.time_base

    descriptions = {}

    print(f"#{i} take: {take_name}, with {len(segments)} segments")
    for seg in segments:
        target_frame_index = (seg["start_frame"] + seg["end_frame"]) // 2
        candidates = []

        seek_point = int(seg["start_time"] / time_base)
        container.seek(seek_point, stream=stream)

        for frame in container.decode(video=0):
            current_time = frame.time
            current_index = ttf(current_time)

            if current_index < seg["start_frame"]:
                continue  # skip until start_frame
            if current_index > seg["end_frame"]:
                break  # segment ends

            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (512, 512), interpolation=RESIZE_METHOD)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. blur check
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            score = laplacian[MASK_BOOL].var()
            if score < LAPLACIAN_THRESHOLD:
                continue

            # 2. duplicate check
            # img_pil = Image.fromarray(gray)
            # cur_hash = imagehash.phash(img_pil)
            dist_to_mid = abs(current_index - target_frame_index)
            candidates.append(
                {
                    "dist": dist_to_mid,
                    "img": img,
                    # "hash": cur_hash,
                    "idx": current_index,
                }
            )

        if not candidates:
            continue

        # select closest to middle frame
        candidates.sort(key=lambda x: x["dist"])
        best_cand = candidates[0]

        filename = f"{take_name}_{best_cand['idx']:06d}_{mod}.jpg"
        save_path = os.path.join(OUT_DIR, filename)
        cv2.imwrite(save_path, best_cand["img"])

        descriptions[f"{take_name}_{best_cand['idx']:06d}"] = {
            "take_name": take_name,
            "step_name": seg["step_name"],
            "step_description": seg["step_description"],
            "frame_index": best_cand["idx"],
            f"image_path_{mod}": save_path,
        }

        del candidates

    container.close()

    # extract slam frames
    for mod in ["slam-left", "slam-right"]:
        rel_path = take_data["frame_aligned_videos"]["aria01"][mod]["relative_path"]
        video_path = os.path.join(root_dir, rel_path)
        container = av.open(video_path)
        # container.streams.video[0].thread_type = "AUTO"
        stream = container.streams.video[0]
        time_base = stream.time_base
        for desc in descriptions.values():
            seek_frame = desc["frame_index"]
            seek_point = int(seek_frame * time_base)
            container.seek(seek_point, stream=stream)
            for frame in container.decode(video=0):
                current_time = frame.time
                current_index = ttf(current_time)
                if current_index < seek_frame:
                    continue
                if current_index > seek_frame:
                    break
                img = frame.to_ndarray(format="bgr24")
                img = cv2.resize(img, (512, 512), interpolation=RESIZE_METHOD)
                filename = f"{take_name}_{current_index:06d}_{mod}.jpg"
                save_path = os.path.join(OUT_DIR, filename)
                cv2.imwrite(save_path, img)
                desc[f"image_path_{mod}"] = save_path
                break

    return descriptions


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # load takes
    with open(TAKES_PATH, "r") as f:
        takes_data = json.load(f)
    valid_takes = [t for t in takes_data if t["take_name"] in GLOBAL_KEYSTEP_LOOKUP]
    valid_takes = [(i, t) for i, t in enumerate(valid_takes)]

    print(f"#workers: {NUM_WORKERS}")
    print(f"#takes: {len(valid_takes)} takes")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        descriptions = list(executor.map(process_single_take, valid_takes))

    all_descriptions = {}
    for desc in descriptions:
        if desc:
            all_descriptions.update(desc)
    out_desc_path = os.path.join(OUT_DIR, "metadata.json")
    with open(out_desc_path, "w") as f:
        json.dump(all_descriptions, f, indent=2)
