import logging
import os

from adapter_medtrinity_demo import MedtrinityDemoAdapter
from filter_resolution import ResolutionFilter
from src.core.pipeline import Pipeline

logger = logging.getLogger("pipeline_logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
)

pipeline = Pipeline(
    logger=logger,
    adapters=[MedtrinityDemoAdapter()],
    filters=[ResolutionFilter(64, 64)],
)

pipeline.scan(allowlist_path="./data/manifest.db", batch_size=1000)
pipeline.build(allowlist_path="./data/manifest.db", output_dir="./data/ready")

os._exit(0)
