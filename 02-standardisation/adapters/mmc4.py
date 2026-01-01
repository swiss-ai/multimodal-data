import asyncio
import concurrent.futures
import glob
import io
import json
import logging
import os
import socket
import zipfile

import aiohttp
from PIL import Image

from pipeline import BaseDataset, ImageSample, SampleMetadata


class MMC4Adapter(BaseDataset):
    def __init__(self, data_dir, batch_size, download_timeout, workers):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.timeout = download_timeout
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
        self.outer_zips = sorted(glob.glob(os.path.join(data_dir, "mmc4_*.zip")))
        if not self.outer_zips:
            raise FileNotFoundError(f"No outer zip files found in {data_dir}")
        raise NotImplementedError("disabled for now")

    @property
    def id(self):
        return "mmc4"

    def iter_documents(self, logger):
        """
        Generator for the JSONL-in-DIR-in-ZIP-in-DIR-in-ZIP logic
        """
        for outer_zip_path in self.outer_zips:
            logger.info(f"Processing outer zip: {outer_zip_path}")

            try:  # outer zip
                with zipfile.ZipFile(outer_zip_path, "r") as outer_z:
                    for inner_filename in outer_z.namelist():
                        if not inner_filename.endswith(".zip"):
                            continue

                        try:  # inner zip
                            inner_zip_bytes = outer_z.read(inner_filename)
                            iob = io.BytesIO(inner_zip_bytes)
                            with zipfile.ZipFile(iob, "r") as inner_z:
                                for jsonl_filename in inner_z.namelist():
                                    if not jsonl_filename.endswith(".jsonl"):
                                        continue

                                    # finally read jsonl
                                    with inner_z.open(jsonl_filename) as f:
                                        text_stream = io.TextIOWrapper(
                                            buffer=f,
                                            encoding="utf-8",
                                        )

                                        # yield each json object
                                        for line in text_stream:
                                            try:
                                                yield json.loads(line)
                                            except json.JSONDecodeError:
                                                continue

                        except zipfile.BadZipFile:
                            logger.warning(f"Corrupt inner zip: {inner_filename}")
            except zipfile.BadZipFile:
                logger.warning(f"Corrupt outer zip: {outer_zip_path}")

    def get_raw_image_url(self, doc):
        return [
            img_meta.get("raw_url")
            for img_meta in doc.get("image_info", [])
            if img_meta.get("raw_url")
        ]

    @staticmethod
    def decode_image_bytes(data):
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode in ("P", "RGBA", "LA"):
                img = img.convert("RGBA")
            return img.convert("RGB")
        except Exception:
            return None

    async def fetch(self, session, url, idx):
        try:
            async with session.get(url, timeout=self.timeout) as response:
                if response.status == 200:
                    data = await response.read()
                    loop = asyncio.get_running_loop()
                    img = await loop.run_in_executor(
                        self.process_pool,
                        self.decode_image_bytes,
                        data,
                    )
                    if img:
                        return img, url, idx
        except Exception:
            pass
        return None

    def stream(self, logger, skip: int | None = None, batch_size: int = 1):
        skip = skip or 0
        logger.info(f"Starting stream with skip={skip}, batch_size={batch_size}")

        def ignore_dns_error(loop, context):
            exception = context.get("exception")
            if isinstance(exception, socket.gaierror):
                return
            msg = context.get("message", "")
            if "Name or service not known" in str(msg) or "gaierror" in str(msg):
                return
            loop.default_exception_handler(context)

        counter = 0
        yield_count = 0

        for file in self.outer_zips:
            logger.info(f"Found outer zip file: {file}")
        logger.info(f"Starting stream. Fetch batch size: {self.batch_size}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(ignore_dns_error)

        async def create_session():
            connector = aiohttp.TCPConnector(
                family=socket.AF_INET,
                limit=0,
                ttl_dns_cache=300,
            )
            # fail fast on dead links
            timeout = aiohttp.ClientTimeout(total=None, connect=5, sock_connect=5)
            return aiohttp.ClientSession(connector=connector, timeout=timeout)

        session = loop.run_until_complete(create_session())
        pending_tasks = set()
        batch = []

        try:
            doc_stream = self.iter_documents(logger)
            doc_iterator = iter(doc_stream)
            documents_exhausted = False

            while True:
                # fill the batch
                while not documents_exhausted and len(pending_tasks) < self.batch_size:
                    try:
                        doc = next(doc_iterator)
                        doc_urls = self.get_raw_image_url(doc)
                        for url in doc_urls:
                            skip -= 1
                            if skip > 0:
                                continue
                            task = loop.create_task(self.fetch(session, url, counter))
                            pending_tasks.add(task)
                            counter += 1
                            if counter % 10000 == 0:
                                logger.debug(
                                    f"Dispatched {counter} fetch tasks, "
                                    f"yielded {yield_count} images."
                                )
                            if len(pending_tasks) >= self.batch_size:
                                break
                    except StopIteration:
                        documents_exhausted = True
                if not pending_tasks:
                    break

                # wait for at least one to compleete
                done, pending_tasks = loop.run_until_complete(
                    asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                )

                # process completed tasks
                # ...and leave pending tasks for NEXT round
                for task in done:
                    try:
                        result = task.result()
                        if result:
                            img, valid_url, valid_id = result
                            yield_count += 1
                            batch.append(
                                ImageSample(
                                    image=img,
                                    meta=SampleMetadata(
                                        dataset_id=self.id,
                                        sample_id=valid_id,
                                        data={"url": valid_url},
                                    ),
                                )
                            )
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                    except Exception:
                        pass

            if batch:
                yield batch

        finally:
            if not session.closed:
                loop.run_until_complete(session.close())
            loop.close()
            logger.info("Finished streaming.")


if __name__ == "__main__":
    data_dir = "/capstor/store/cscs/swissai/infra01/medical/raw/MMC4"
    adapter = MMC4Adapter(
        data_dir=data_dir,
        batch_size=1000,
        download_timeout=3,
        workers=100,
    )

    logger = logging.getLogger("mmc4_adapter")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    counter = 0
    for batch in adapter.stream(logger, skip=0):
        for sample in batch:
            counter += 1
            if counter % 1000 == 0:
                logger.info(f"Sample {counter}: {sample.meta.data['url']}")
