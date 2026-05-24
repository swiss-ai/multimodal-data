import os
import shutil

import av
import cv2
import numpy as np

# original = (1408, 1408)
sizes = [1408, 1024, 768, 512, 384, 256]


if __name__ == "__main__":
    map_x = np.load("aria_map_x.npy")
    map_y = np.load("aria_map_y.npy")

    root_dir = os.environ.get("EGOEXO4D_TAKES_DIR", "")
    paths = [os.path.join(root_dir, take) for take in os.listdir(root_dir)]
    paths.sort()

    counter = 0
    counter2 = 0

    shutil.rmtree("sample", ignore_errors=True)
    os.makedirs("sample/original", exist_ok=True)
    for size in sizes:
        os.makedirs(f"sample/{size}", exist_ok=True)

    for i in range(0, len(paths), 250):
        take = paths[i]
        av_path = os.path.join(take, "frame_aligned_videos", "aria01_214-1.mp4")
        if not os.path.exists(av_path):
            continue

        with av.open(av_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            seek_point = int(stream.duration * 0.5)
            container.seek(seek_point, stream=stream)

            for i, frame in enumerate(container.decode(video=0), start=0):
                img_orig = frame.to_ndarray(format="bgr24")
                img = cv2.remap(img_orig, map_x, map_y, interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                score = laplacian.var()
                score += i // 10
                if score < 30:
                    print(score)
                    continue

                print(take)
                cv2.imwrite(f"sample/original/{counter2:02}.png", img_orig)

                for size in sizes:
                    if size == 1408:
                        resized = img
                    else:
                        resized = cv2.resize(
                            img,
                            (size, size),
                            interpolation=cv2.INTER_LANCZOS4,
                        )
                    cv2.imwrite(f"sample/{size}/{counter2:02}.png", resized)

                counter += 1
                counter2 += 1
                break
