import os
import random
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets import load_dataset

OUT_ROOT = "./ytc"
# Languages for download: LANGS = ["es", "nl", "en", "fr", "it", "pt"]
LANGS = []

MAX_WORKERS = 6


def ytc_download(ytid, temp):
    return subprocess.run(
        [
            "yt-dlp",
            "--cookies",
            "cookies.txt",
            "--js-runtimes",
            "node",
            "--remote-components",
            "ejs:github",
            "--sleep-interval",
            "2",
            "--max-sleep-interval",
            "5",
            "-f",
            "bestaudio",
            "--no-playlist",
            "-o",
            temp,
            f"https://www.youtube.com/watch?v={ytid}",
        ],
        check=False,
    )


def process_one_video(args):
    out_dir, ytid, segments = args

    if all(os.path.exists(os.path.join(out_dir, file)) for file, _, _ in segments):
        return

    print("Downloading", ytid)
    temp = os.path.join(out_dir, f"__tmp_{ytid}_{os.getpid()}.m4a")

    ytc_download(ytid, temp)

    if not os.path.exists(temp):
        print("Download failed:", ytid)
        time.sleep(random.uniform(1, 2))
        return

    for filename, start, duration in segments:
        out_path = os.path.join(out_dir, filename)
        if os.path.exists(out_path):
            continue

        # start is in milisec, duration in centisec.
        start_sec = start / 1000.0
        dur_sec = duration / 100.0

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_sec),
                "-t",
                str(dur_sec),
                "-i",
                temp,
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                out_path,
            ],
            check=False,
        )

        time.sleep(0.05)

    try:
        os.remove(temp)
    except FileNotFoundError:
        pass

    time.sleep(random.uniform(1, 2))


def main():
    for lang in LANGS:
        print("Processing", lang)

        out_dir = os.path.join(OUT_ROOT, lang)
        os.makedirs(out_dir, exist_ok=True)

        ds = load_dataset("nvidia/Granary", f"{lang}_ytc", split="asr")

        videos = defaultdict(list)
        for d in ds:
            filename = d["audio_filepath"].split("/")[-1]
            base = filename.removesuffix(".wav")
            ytid, start, duration = base.rsplit("-", 2)
            videos[ytid].append((filename, int(start), int(duration)))

        print("Num of unique videos (audios):", len(videos))

        jobs = [(out_dir, ytid, segments) for ytid, segments in videos.items()]

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(process_one_video, j) for j in jobs]
            for _ in as_completed(futures):
                pass

    print("\nDone.")


if __name__ == "__main__":
    main()
