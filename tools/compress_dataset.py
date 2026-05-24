import argparse
import io
import multiprocessing
import os
import shutil
import subprocess
import tarfile
import tempfile

import numpy as np
from loguru import logger


def get_tar_files(directory, sort=False):
    tar_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tar"):
                tar_files.append(os.path.join(root, file))
    if sort:
        tar_files.sort()
    return tar_files


def verify_tar(tar_path):
    res = subprocess.run(
        ["tar", "-tf", tar_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if res.returncode != 0:
        return False
    return True


def process_file(args):
    in_path, tmp_path, out_path = args

    if not in_path.endswith(".tar"):
        shutil.copy2(in_path, out_path)
        return True

    with (
        tarfile.open(in_path, "r|") as in_tar,
        tarfile.open(tmp_path, "w|") as tmp_tar,
    ):
        for member in in_tar:
            file_obj = in_tar.extractfile(member)
            assert file_obj is not None

            if not member.name.endswith(".jpg"):
                tmp_tar.addfile(member, file_obj)
                continue

            fd_in, temp_in = tempfile.mkstemp(dir="/dev/shm", suffix=".jpg")
            fd_out, temp_out = tempfile.mkstemp(dir="/dev/shm", suffix=".jxl")

            img_bytes = file_obj.read()
            os.write(fd_in, img_bytes)
            os.close(fd_in)
            os.close(fd_out)

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            res = subprocess.run(
                ["cjxl", "-q", "100", temp_in, temp_out],
                capture_output=True,
                env=env,
            )
            assert res.returncode == 0

            with open(temp_out, "rb") as f:
                jxl_bytes = f.read()
            if len(jxl_bytes) < len(img_bytes):
                member.name = member.name[:-4] + ".jxl"
                member.size = len(jxl_bytes)
                tmp_tar.addfile(member, io.BytesIO(jxl_bytes))
            else:
                tmp_tar.addfile(member, io.BytesIO(img_bytes))

            os.remove(temp_in)
            os.remove(temp_out)

    if not verify_tar(tmp_path):
        logger.error(f"Verification failed for {tmp_path}. Removing it.")
        os.remove(tmp_path)
        return False

    shutil.move(tmp_path, out_path)

    return True


def main():
    parser = argparse.ArgumentParser(description="WebDataset JXL Compressor")
    parser.add_argument("--ncpus", type=int, required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--source-dir", type=str, required=True)
    parser.add_argument("--dest-dir", type=str, required=True)
    args = parser.parse_args()

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    all_files = get_tar_files(args.source_dir, sort=True)
    work_files = np.array_split(all_files, args.task_count)[args.task_id]
    if len(work_files) == 0:
        logger.warning("No files assigned to task. Exiting.")

    logger.info(f"Assigned {len(work_files)} files.")
    logger.info(f"{work_files[0]}..{work_files[-1]}")

    tasks = []
    for src_path in work_files:
        rel_path = os.path.relpath(src_path, args.source_dir)
        dest_path = os.path.join(args.dest_dir, rel_path)
        tmp_path = dest_path + ".tmp"

        assert os.path.isfile(src_path) and src_path.endswith(".tar")
        if os.path.exists(dest_path):
            logger.warning(f"{dest_path} exists. Skipping {src_path}.")
            continue
        if os.path.exists(tmp_path):
            logger.warning(f"{tmp_path} already exists. Removing now.")
            os.remove(tmp_path)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        tasks.append((src_path, tmp_path, dest_path))

    with multiprocessing.Pool(args.ncpus) as pool:
        results = pool.map(process_file, tasks)

    for success in results:
        if not success:
            logger.error("Processing failed for one of the files.")


if __name__ == "__main__":
    main()
