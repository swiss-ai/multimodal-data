# python3
# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Open Images image downloader.

This script downloads a subset of Open Images images, given a list of image ids.
Typical uses of this tool might be downloading images:
- That contain a certain category.
- That have been annotated with certain types of annotations (e.g. Localized
Narratives, Exhaustively annotated people, etc.)

The input file IMAGE_LIST should be a text file containing one image per line
with the format <SPLIT>/<IMAGE_ID>, where <SPLIT> is either "train", "test",
"validation", or "challenge2018"; and <IMAGE_ID> is the image ID that uniquely
identifies the image in Open Images. A sample file could be:
  train/f9e0434389a1d4dd
  train/1a007563ebc18664
  test/ea8bfd4e765304db

"""

import argparse
import os
import re
import sys
import time
from concurrent import futures

import boto3
import botocore
import tqdm

BUCKET_NAME = "open-images-dataset"
REGEX = r"(test|train|validation|challenge2018)/([a-fA-F0-9]*)"


def check_and_homogenize_one_image(image):
    split, image_id = re.match(REGEX, image).groups()
    yield split, image_id


def check_and_homogenize_image_list(image_list):
    for line_number, image in enumerate(image_list):
        try:
            yield from check_and_homogenize_one_image(image)
        except (ValueError, AttributeError):
            raise ValueError(
                f"ERROR in line {line_number} of the image list. The following image "
                f'string is not recognized: "{image}".'
            )


def read_image_list_file(image_list_file):
    with open(image_list_file, "r") as f:
        for line in f:
            yield line.strip().replace(".jpg", "")


def count_images_in_list(image_list_file):
    with open(image_list_file, "r") as f:
        return sum(1 for line in f if line.strip())


def cleanup_failed_download(destination):
    try:
        if os.path.exists(destination):
            os.remove(destination)
    except OSError:
        pass


def is_missing_object_error(exception):
    if not isinstance(exception, botocore.exceptions.ClientError):
        return False

    response = exception.response or {}
    error = response.get("Error", {})
    code = str(error.get("Code", ""))
    status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status_code == 404


def download_one_image(
    bucket,
    split,
    image_id,
    download_folder,
    retries,
    retry_backoff,
    skip_existing,
):
    destination = os.path.join(download_folder, f"{image_id}.jpg")

    if skip_existing and os.path.exists(destination) and os.path.getsize(destination) > 0:
        return {
            "status": "skipped",
            "image": f"{split}/{image_id}",
            "message": None,
        }

    for attempt in range(retries + 1):
        try:
            bucket.download_file(f"{split}/{image_id}.jpg", destination)
            return {
                "status": "downloaded",
                "image": f"{split}/{image_id}",
                "message": None,
            }
        except (
            botocore.exceptions.BotoCoreError,
            botocore.exceptions.ClientError,
            OSError,
        ) as exception:
            cleanup_failed_download(destination)
            if is_missing_object_error(exception):
                return {
                    "status": "missing",
                    "image": f"{split}/{image_id}",
                    "message": None,
                }
            if attempt == retries:
                return {
                    "status": "failed",
                    "image": f"{split}/{image_id}",
                    "message": str(exception),
                }
            time.sleep(retry_backoff * (2**attempt))


def drain_completed_futures(pending_futures, progress_bar, failures, stats):
    done_futures, pending_futures = futures.wait(pending_futures, return_when=futures.FIRST_COMPLETED)
    for future in done_futures:
        result = future.result()
        stats[result["status"]] += 1
        if result["status"] == "failed":
            failures.append(result)
        progress_bar.update(1)
    return pending_futures


def download_all_images(args):
    """Downloads all images specified in the input file."""
    bucket = boto3.resource("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)).Bucket(
        BUCKET_NAME
    )

    download_folder = args["download_folder"] or os.getcwd()

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    try:
        image_list = check_and_homogenize_image_list(read_image_list_file(args["image_list"]))
    except ValueError as exception:
        sys.exit(exception)

    progress_bar = tqdm.tqdm(
        total=count_images_in_list(args["image_list"]),
        desc="Downloading images",
        leave=True,
    )
    max_queue_size = args["max_queue_size"] or args["num_processes"] * 4
    failures = []
    stats = {"downloaded": 0, "failed": 0, "missing": 0, "skipped": 0}
    with futures.ThreadPoolExecutor(max_workers=args["num_processes"]) as executor:
        pending_futures = set()
        for split, image_id in image_list:
            pending_futures.add(
                executor.submit(
                    download_one_image,
                    bucket,
                    split,
                    image_id,
                    download_folder,
                    args["retries"],
                    args["retry_backoff"],
                    args["skip_existing"],
                )
            )
            while len(pending_futures) >= max_queue_size:
                pending_futures = drain_completed_futures(pending_futures, progress_bar, failures, stats)
        while pending_futures:
            pending_futures = drain_completed_futures(pending_futures, progress_bar, failures, stats)
    progress_bar.close()

    if failures and args["failed_list"]:
        with open(args["failed_list"], "w") as f:
            for failure in failures:
                f.write(f"{failure['image']}\n")
    elif args["failed_list"] and os.path.exists(args["failed_list"]):
        os.remove(args["failed_list"])

    print(
        (
            f"Finished: downloaded={stats['downloaded']} "
            f"skipped={stats['skipped']} missing={stats['missing']} failed={stats['failed']}"
        ),
        file=sys.stderr,
    )
    if failures:
        first_failure = failures[0]
        sys.exit(
            "ERROR: "
            f"{len(failures)} image(s) failed. "
            f"First failure: {first_failure['image']} -> {first_failure['message']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "image_list",
        type=str,
        default=None,
        help=("Filename that contains the split + image IDs of the images to download. Check the document"),
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=5,
        help="Number of parallel processes to use (default is 5).",
    )
    parser.add_argument(
        "--max_queue_size",
        type=int,
        default=0,
        help=("Maximum number of in-flight downloads to keep queued. Defaults to 4x --num_processes."),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Number of retries for transient download failures (default is 6).",
    )
    parser.add_argument(
        "--retry_backoff",
        type=float,
        default=1.0,
        help="Initial retry backoff in seconds; doubles after each attempt.",
    )
    parser.add_argument(
        "--download_folder",
        type=str,
        default=None,
        help="Folder where to download the images.",
    )
    parser.add_argument(
        "--failed_list",
        type=str,
        default=None,
        help="Optional file where failed image identifiers will be written.",
    )
    parser.add_argument(
        "--skip_existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip images whose target file already exists and is non-empty.",
    )
    download_all_images(vars(parser.parse_args()))
