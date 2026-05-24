# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of NVIDIA CORPORATION nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY NVIDIA CORPORATION AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Disclaimer:
For each dataset an user elects to use, the user is responsible for checking if the dataset license
is fit for the intended purpose.
"""

import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Literal

try:
    import click
    import pymupdf
    import requests
    from PIL import Image
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please install the dependencies using the following command:")
    print("pip install click pillow pymupdf requests")
    exit(1)

try:
    from parallel import ProcessBound, process_generator, thread_generator
except ImportError:
    print("Error: parallel.py not found in path")
    print("Make sure parallel.py is in the same directory or in PYTHONPATH")
    exit(1)

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class HTTPRangeReader(io.RawIOBase):
    """A class to read files from HTTP(S) URLs without downloading the whole file."""

    url: str
    final_url: str | None
    file_size: int
    pos: int
    _session: requests.Session | None
    _closed: bool

    total_bytes_read: int = 0
    total_num_requests: int = 0

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self._session = requests.Session()
        self._session.headers.update({"Connection": "keep-alive"})
        head = self._session.head(self.url, allow_redirects=True)
        self.total_num_requests += 1
        try:
            head.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                head = self._session.get(self.url, allow_redirects=True, stream=True)
                head.raise_for_status()
                self.final_url = head.url
            else:
                raise e
        self.final_url = None
        self.file_size = int(head.headers.get("Content-Length", 0))
        self._closed = False

    def suspend(self) -> None:
        """Close the HTTP connection, allowing to reconnect when needed. Afterwards, no resources are used."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def resume(self) -> None:
        """Reopen the HTTP connection to retrieve more data."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({"Connection": "keep-alive"})

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.file_size + offset
        else:
            raise ValueError("Invalid value for whence")
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _session_get(self, range_start: int, range_end: int, stream: bool = False) -> requests.Response:
        for _retry in range(2):
            url = self.url
            if self.final_url is not None:
                url = self.final_url
            else:
                url = self.url
            headers = {"Range": f"bytes={range_start}-{range_end}"}
            resp = self._session.get(url, headers=headers, stream=stream)
            self.total_num_requests += 1
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403 and self.final_url is not None:
                    # Retry to resolve the final url again.
                    self.final_url = None
                    continue
                raise e
            if self.final_url is None:
                self.final_url = resp.url
            return resp

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        if size == -1:
            size = self.file_size - self.pos
        if self.pos >= self.file_size:
            return b""
        end = min(self.pos + size - 1, self.file_size - 1)
        resp = self._session_get(self.pos, end)
        data = resp.content
        read_len = len(data)
        self.pos += read_len
        self.total_bytes_read += read_len
        return data

    def readinto(self, b: bytearray) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file.")
        size = len(b)
        if self.pos >= self.file_size:
            return 0
        end = min(self.pos + size - 1, self.file_size - 1)
        resp = self._session_get(self.pos, end, stream=True)
        n = 0
        for chunk in resp.iter_content(chunk_size=8192):
            chunk_len = min(len(chunk), size - n)
            b[n : n + chunk_len] = chunk[:chunk_len]
            n += chunk_len
            if n >= size:
                break
        self.pos += n
        self.total_bytes_read += n
        return n

    def close(self) -> None:
        self._closed = True
        if self._session is not None:
            self._session.close()
            self._session = None
        super().close()

    @property
    def closed(self) -> bool:
        return self._closed

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


class KeepAliveZipReader:
    """A class to read files from zip archives located at HTTP(S) URLs without downloading the whole zip.
    Keeps the zip archive open in memory to avoid re-downloading the zip archive index. Does not use any other resources
    between downloads. Allows for fast retrieval of multiple files from the same zip archive.
    Thread-safe for parallel downloads.
    """

    def __init__(self):
        self.zip_handles = {}
        self.zip_locks = defaultdict(Lock)
        self.main_lock = Lock()

    def download_file(self, zip_url: str, file_in_zip: str, output_path: Path) -> None:
        """Download a file from a zip archive located at a HTTP(S) URL and save it to `output_path`, without downloading the whole zip.
        Closes the HTTP connection after downloading the file, but keeps the zip index open in memory for more data retrieval.
        Thread-safe for parallel downloads from different zip archives."""
        with self.zip_locks[zip_url]:
            with self.main_lock:
                if zip_url not in self.zip_handles:
                    http_reader = HTTPRangeReader(zip_url)
                    zip_reader = zipfile.ZipFile(io.BufferedReader(http_reader, buffer_size=5 * 1024 * 1024))
                    self.zip_handles[zip_url] = (http_reader, zip_reader)
                else:
                    http_reader, zip_reader = self.zip_handles[zip_url]

            http_reader.resume()
            tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
            try:
                with (
                    zip_reader.open(file_in_zip) as source,
                    open(tmp_path, "wb") as target,
                ):
                    while True:
                        chunk = source.read(8192)
                        if not chunk:
                            break
                        target.write(chunk)
                tmp_path.rename(output_path)
            except Exception as e:
                tmp_path.unlink(missing_ok=True)
                with self.main_lock:
                    if zip_url in self.zip_handles:
                        zip_reader.close()
                        http_reader.close()
                        del self.zip_handles[zip_url]
                raise e
            finally:
                http_reader.suspend()


@click.group()
def cli():
    """Tool for downloading CCPDF dataset files."""
    pass


@cli.group()
def zip():
    """Operations on zip archives located at HTTP(S) URLs without downloading the whole zip."""
    pass


@zip.command("list")
@click.argument("zip_url", type=str)
def zip_list(zip_url: str) -> None:
    """List all files in a zip archive located at a HTTP(S) URL without downloading the whole zip."""
    with HTTPRangeReader(zip_url) as reader:
        with zipfile.ZipFile(io.BufferedReader(reader, buffer_size=5 * 1024 * 1024)) as zf:
            print(f"Files in {zip_url}:")
            for filename in zf.namelist():
                print(f"  {filename}")


@zip.command("extract")
@click.argument("zip_url", type=str)
@click.argument("file_in_zip", type=str)
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("-v", "--verbose", is_flag=True)
def zip_extract(zip_url: str, file_in_zip: str, output_path: Path, verbose: bool) -> None:
    """Extract a file from a zip archive located at a HTTP(S) URL and save it to OUTPUT_PATH, without downloading the whole zip."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with HTTPRangeReader(zip_url) as reader:
            with zipfile.ZipFile(io.BufferedReader(reader, buffer_size=5 * 1024 * 1024)) as zf:
                if file_in_zip not in zf.namelist():
                    raise FileNotFoundError(f"{file_in_zip} not found in the zip archive.")
                with zf.open(file_in_zip) as source, open(tmp_path, "wb") as target:
                    while True:
                        chunk = source.read(8192)
                        if not chunk:
                            break
                        target.write(chunk)
            if verbose:
                print(f"Requests: {reader.total_num_requests}", file=sys.stderr)
                print(f"Bytes read: {reader.total_bytes_read}", file=sys.stderr)
        tmp_path.rename(output_path)
        if verbose:
            print(f"Extracted {zip_url}/{file_in_zip} to {output_path}", file=sys.stderr)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


_re_ccpdf_filename = re.compile(r"^(\d{7})_(\d+)\.(png|jpg)$")
_re_ccpdf_path = re.compile(r"^\d{4}/(\d{7})_pdf/(\d+)\.png$")


def _parse_image_filename(image_name: str) -> tuple[str, int, str] | None:
    """Parse image filename/path to extract PDF info.

    Supports two formats:
      - Flat:  {pdf_name}_{page_number}.{ext}       e.g. "0021642_2.png"
      - Path:  {prefix}/{pdf_name}_pdf/{page}.png   e.g. "0000/0000000_pdf/3.png"

    Returns:
        Tuple of (pdf_filename, page_number, zip_url) or None if parsing fails
    """
    match = _re_ccpdf_filename.match(image_name) or _re_ccpdf_path.match(image_name)
    if not match:
        return None

    pdf_name = match.group(1)
    page_number = int(match.group(2))
    url = f"https://digitalcorpora.s3.amazonaws.com/corpora/files/CC-MAIN-2021-31-PDF-UNTRUNCATED/zipfiles/{pdf_name[0]}000-{pdf_name[0]}999/{pdf_name[:4]}.zip"

    return (f"{pdf_name}.pdf", page_number, url)


def _extract_image_metadata(image_name: str) -> tuple[str, str, int, str]:
    """Parse and extract metadata from an image filename.

    Args:
        image_name: Image filename to parse

    Returns:
        Tuple of (image_name, pdf_path, page_number, url)
    """
    parsed = _parse_image_filename(image_name)
    assert parsed is not None, f"Failed to parse image filename: {image_name}"
    pdf_path, page_number, url = parsed
    return (image_name, pdf_path, page_number, url)


def extract_ocr_format_metadata(sample: dict) -> list[tuple[str, str, int, str | None]]:
    """Extract metadata from OCR format JSONL.

    Returns:
        List of tuples (image_name, pdf_path, page_number, url)
    """
    image_name = sample.get("image")
    if image_name is None:
        return []
    return [_extract_image_metadata(image_name)]


def extract_conversation_format_metadata(
    sample: dict,
) -> list[tuple[str, str, int, str | None]]:
    """Extract metadata from conversation format JSONL.

    Returns:
        List of tuples (image_name, pdf_path, page_number, url)
    """
    results = []
    for message in sample.get("messages", []):
        for fragment in message.get("content", []):
            if isinstance(fragment, dict) and fragment.get("type") == "image":
                results.append(_extract_image_metadata(fragment["image"]))
    return results


def _wrap_iterator(
    iterator,
    workers: int,
    progress: bool,
    total: int,
    desc: str,
    parallel: Literal["thread", "process"] = "process",
):
    """Wrap iterator with optional threading and progress bar.

    Args:
        iterator: The base iterator
        workers: Number of workers (>1 for parallel execution)
        progress: Whether to show progress bar
        total: Total number of items
        desc: Progress bar description

    Returns:
        Wrapped iterator
    """
    if workers > 1:
        if parallel == "thread":
            iterator = thread_generator(iterator, pool_size=workers)
        elif parallel == "process":
            iterator = process_generator(iterator, pool_size=workers)
    if progress and TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=total, desc=desc)
    return iterator


def download_pdfs_from_zip(
    zip_url: str,
    pdf_paths: list[str],
    pdf_download_dir: Path,
    zip_reader: KeepAliveZipReader,
    verbose: bool,
) -> tuple[int, int]:
    """Download all needed PDFs from a single zip. Returns (success_count, error_count)."""
    success, errors = 0, 0
    for pdf_path in pdf_paths:
        pdf_file = pdf_download_dir / pdf_path
        if pdf_file.exists():
            success += 1
            continue
        pdf_tmp = pdf_file.with_suffix(pdf_file.suffix + ".tmp")
        pdf_tmp.unlink(missing_ok=True)
        try:
            pdf_file.parent.mkdir(parents=True, exist_ok=True)
            zip_reader.download_file(zip_url, pdf_path, pdf_file)
            if verbose:
                print(f"Downloaded {pdf_path}", file=sys.stderr)
            success += 1
        except Exception as e:
            if verbose:
                print(f"Error downloading {pdf_path}: {e}", file=sys.stderr)
            errors += 1
    return success, errors


def render_page_to_png(
    image_name: str,
    pdf_path: str,
    page_number: int,
    output_dir: Path,
    pdf_download_dir: Path,
    verbose: bool,
    output_image_max_size: tuple[int, int] = (1024, 1280),
) -> bool:
    """Render a PDF page to PNG. Returns True if successful, False otherwise."""
    pdf_file = pdf_download_dir / pdf_path
    pdf_page_path = output_dir / image_name

    # Check if image already exists
    if pdf_page_path.exists():
        if verbose:
            print(f"Image {image_name} already exists", file=sys.stderr)
        return True

    # Clean up leftover tmp files from interrupted downloads
    image_tmp = pdf_page_path.with_suffix(pdf_page_path.suffix + ".tmp")
    image_tmp.unlink(missing_ok=True)

    # Check if PDF exists
    if not pdf_file.exists():
        if verbose:
            print(f"PDF {pdf_path} not found", file=sys.stderr)
        return False

    try:
        doc = pymupdf.Document(pdf_file)
        page = doc.load_page(page_number)

        zoom = min(
            output_image_max_size[0] / page.rect.width,
            output_image_max_size[1] / page.rect.height,
        )

        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        assert img.size[0] <= output_image_max_size[0] and img.size[1] <= output_image_max_size[1], (
            f"Image size {img.size} exceeds max size {output_image_max_size}, rect={page.rect}, zoom={zoom}"
        )

        tmp_path = pdf_page_path.with_suffix(pdf_page_path.suffix + ".tmp")
        pdf_page_path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(
                f"Saving image ({img.size[0]}x{img.size[1]}) to {pdf_page_path}",
                file=sys.stderr,
            )
        img.save(tmp_path, format="PNG")
        tmp_path.rename(pdf_page_path)
        return True
    except Exception as e:
        image_tmp.unlink(missing_ok=True)
        if verbose:
            print(f"Error rendering {image_name}: {e}", file=sys.stderr)
        return False


@cli.command("download")
@click.argument("jsonl_file", type=click.Path(path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.argument("pdf_download_dir", type=click.Path(path_type=Path))
@click.option("-v", "--verbose", is_flag=True)
@click.option(
    "--format",
    "data_format",
    type=click.Choice(["ocr", "conversation"]),
    required=True,
    help="Input JSONL format",
)
@click.option("--progress", is_flag=True, help="Show progress bar (requires tqdm)")
@click.option("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
def download_from_metadata(
    jsonl_file: Path,
    output_dir: Path,
    pdf_download_dir: Path,
    verbose: bool,
    data_format: str,
    progress: bool,
    workers: int,
) -> None:
    """Download PDF files from a JSONL file containing metadata, then render pages to PNGs."""
    if verbose:
        print(
            f"Downloading PDF files from {jsonl_file} to {output_dir} and {pdf_download_dir}",
            file=sys.stderr,
        )
        print(f"Input format: {data_format}", file=sys.stderr)
        print(f"Using {workers} worker(s)", file=sys.stderr)

    if progress and not TQDM_AVAILABLE:
        print(
            "Warning: tqdm not available. Install with: pip install tqdm",
            file=sys.stderr,
        )
        print("Continuing without progress bar...", file=sys.stderr)

    if workers < 1:
        print("Error: --workers must be at least 1", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_download_dir.mkdir(parents=True, exist_ok=True)

    zip_reader = KeepAliveZipReader()

    # Choose extraction function based on format
    if data_format == "ocr":
        extract_metadata = extract_ocr_format_metadata
    else:
        extract_metadata = extract_conversation_format_metadata

    # Phase 1: Collect all unique PDF files and render tasks
    if verbose:
        print("\n=== Phase 1: Collecting tasks ===", file=sys.stderr)

    pdf_files = {}
    render_tasks = []
    with open(jsonl_file, "r") as f:
        for line in f:
            sample = json.loads(line)
            metadata_list = extract_metadata(sample)
            for image_name, pdf_path, page_number, url in metadata_list:
                if pdf_path not in pdf_files:
                    pdf_files[pdf_path] = url
                render_tasks.append((image_name, pdf_path, page_number))

    if verbose:
        print(f"Found {len(pdf_files)} unique PDFs to download", file=sys.stderr)
        print(f"Found {len(render_tasks)} pages to render", file=sys.stderr)

    # Phase 2: Download all PDFs, grouped by zip to minimise connections
    if verbose:
        print("\n=== Phase 2: Downloading PDFs ===", file=sys.stderr)

    # Group PDFs by their source zip URL
    pdfs_by_zip: dict[str, list[str]] = defaultdict(list)
    for pdf_path, url in pdf_files.items():
        if url:
            pdfs_by_zip[url].append(pdf_path)

    pdf_success_count = 0
    pdf_error_count = 0
    iterator = (
        ProcessBound(
            download_pdfs_from_zip,
            zip_url,
            pdf_list,
            pdf_download_dir,
            zip_reader,
            verbose,
        )
        for zip_url, pdf_list in pdfs_by_zip.items()
    )
    iterator = _wrap_iterator(
        iterator,
        workers,
        progress,
        len(pdfs_by_zip),
        "Downloading zips",
        parallel="thread",
    )
    for success, errors in iterator:
        pdf_success_count += success
        pdf_error_count += errors

    if verbose:
        print("\nPDF download complete:", file=sys.stderr)
        print(f"  Successful: {pdf_success_count}", file=sys.stderr)
        print(f"  Errors: {pdf_error_count}", file=sys.stderr)

    # Phase 3: Render all pages to PNGs
    if verbose:
        print("\n=== Phase 3: Rendering pages to PNGs ===", file=sys.stderr)

    render_success_count = 0
    render_error_count = 0
    iterator = (
        ProcessBound(
            render_page_to_png,
            image_name,
            pdf_path,
            page_number,
            output_dir,
            pdf_download_dir,
            verbose,
        )
        for image_name, pdf_path, page_number in render_tasks
    )
    iterator = _wrap_iterator(
        iterator,
        workers,
        progress,
        len(render_tasks),
        "Rendering pages",
        parallel="process",
    )
    for success in iterator:
        if success:
            render_success_count += 1
        else:
            render_error_count += 1

    # Summary
    if verbose or pdf_error_count > 0 or render_error_count > 0:
        print("\n=== Summary ===", file=sys.stderr)
        print(f"PDFs downloaded: {pdf_success_count}", file=sys.stderr)
        print(f"Pages rendered: {render_success_count}", file=sys.stderr)
        if pdf_error_count > 0 or render_error_count > 0:
            print(f"Total errors: {pdf_error_count + render_error_count}", file=sys.stderr)


if __name__ == "__main__":
    cli()
