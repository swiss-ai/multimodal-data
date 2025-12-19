import logging
import os
import shutil

from adapters.medtrinity_demo import MedtrinityDemoAdapter
from filters.deduplication import ImageDeduplication
from filters.resolution import ResolutionFilter
from src.pipeline import Pipeline

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
)

if os.path.exists("data/"):
    shutil.rmtree("data/")  # fresh start

pipeline = Pipeline(
    logger=logger,
    datasets=[
        MedtrinityDemoAdapter(),
    ],
    filters=[
        ResolutionFilter(64, 64),
        ImageDeduplication(db_path="./data/dedup.db", algorithm="phash"),
    ],
    allowlist_path="./data/manifest.db",
    batch_size=1000,
    output_dir="./data/ready",
)

pipeline.scan()
pipeline.build()

os._exit(0)
