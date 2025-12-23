import logging
import os
import shutil
import warnings

import datasets

from adapters.medtrinity_demo import MedtrinityDemoAdapter
from filters.deduplication import ImageDeduplication
from filters.resolution import ResolutionFilter
from src.logging import setup_logging
from src.pipeline import Pipeline

datasets.disable_progress_bar()
warnings.simplefilter(action="ignore", category=FutureWarning)


DATA_DIR = "./data"
FRESH_START = False


logger = setup_logging(
    level=logging.DEBUG,
    log_file=f"{DATA_DIR}/pipeline.log",
)

if FRESH_START:
    logger.warning("Performing fresh start, deleting existing data directory")
    shutil.rmtree(DATA_DIR, ignore_errors=True)

adapters = [
    MedtrinityDemoAdapter(),
]

filters = [
    ResolutionFilter(64, 64),
    ImageDeduplication(f"{DATA_DIR}/dedup.db", "phash"),
]

pipeline = Pipeline(
    datasets=adapters,
    filters=filters,
    sinks=None,
    data_dir=DATA_DIR,
    num_workers=8,
    batch_size=500,
)

pipeline.scan()

os._exit(0)
