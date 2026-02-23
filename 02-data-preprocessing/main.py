import json
import logging
import os
import sys
import time
from functools import partial

from adapters import ADAPTER_REGISTRY
from filters import FILTER_REGISTRY
from pipeline import Pipeline, setup_logging
from writers import HuggingFaceDatasetWriter, WebDatasetWriter

WRITER_REGISTRY = {
    "huggingface": HuggingFaceDatasetWriter,
    "webdataset": WebDatasetWriter,
}


def build_factories(config: list[dict], registry: dict[str, type]) -> list:
    factories = []
    for cfg in config:
        cfg = cfg.copy()
        adapter_id = cfg.pop("id")
        if adapter_id not in registry:
            print("legal adapters:", registry.keys())
            raise ValueError(f"Unknown adapter/filter id: {adapter_id}")
        factory = registry[adapter_id]
        factories.append(partial(factory, **cfg))
    return factories


def main(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    adapters_config = config["adapters"]
    filters_config = config.get("filters", [])
    writer_config = config["writer"]

    log_file = config["pipeline"]["log_file"]
    manifest_db = config["pipeline"]["manifest_db"]
    checkpoint_db = config["pipeline"]["checkpoint_db"]
    num_workers = config["pipeline"]["num_workers"]
    batch_size = config["pipeline"]["batch_size"]

    setup_logging(level=logging.DEBUG, log_file=log_file)
    logger = logging.getLogger("pipeline")
    logger.debug("Starting pipeline with config: %s", config)

    adapter_factories = build_factories(adapters_config, ADAPTER_REGISTRY)
    filter_factories = build_factories(filters_config, FILTER_REGISTRY)
    writer_config = writer_config.copy()
    writer_type = writer_config.pop("type", "huggingface")
    if writer_type not in WRITER_REGISTRY:
        raise ValueError(
            f"Unknown writer type: {writer_type}. Options: {list(WRITER_REGISTRY.keys())}"
        )
    writer = WRITER_REGISTRY[writer_type](**writer_config)

    pipeline = Pipeline(
        datasets=[f() for f in adapter_factories],
        filter_factories=filter_factories,
        writer=writer,
        manifest_db_path=manifest_db,
        checkpoint_db_path=checkpoint_db,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    pipeline.scan()

    time.sleep(10)
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
