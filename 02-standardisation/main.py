import logging
import os

# adapters
from adapters.medtrinity_demo import MedtrinityDemoAdapter

# filters
from filters.deduplication import ImageDeduplication
from filters.resolution import ResolutionFilter

# core pipeline
from pipeline import Pipeline, WebDatasetSink, setup_logging

DATA_DIR = "./data"
setup_logging(level=logging.DEBUG, log_file=f"{DATA_DIR}/pipeline.log")

adapters = [
    MedtrinityDemoAdapter(),
]

filters = [
    ResolutionFilter(64, 64),
    ImageDeduplication(f"{DATA_DIR}/dedup.db", "phash"),
]

sink = WebDatasetSink(
    output_dir=f"{DATA_DIR}/webdataset",
    samples_per_shard=100,  # TODO: increase for prod
    target_shard_bytes=500_000_000,
)

pipeline = Pipeline(
    datasets=adapters,
    filters=filters,
    sinks=sink,
    data_dir=DATA_DIR,
    num_workers=8,
    batch_size=500,
)

pipeline.scan()

os._exit(0)
