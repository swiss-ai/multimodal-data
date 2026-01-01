import json
import logging
import os
import sys
import time
from functools import partial

from adapters import ADAPTER_REGISTRY
from filters import FILTER_REGISTRY
from pipeline import Pipeline, setup_logging


def build_factories(config: list[dict], registry: dict[str, type]) -> list:
    factories = []
    for cfg in config:
        cfg = cfg.copy()
        factory = registry[cfg.pop("id")]
        factories.append(partial(factory, **cfg))
    return factories


def main(config_path: str):
    with open(config_path) as f:
        config = json.load(f)

    setup_logging(level=logging.DEBUG, log_file=config["pipeline"]["log_file"])
    logger = logging.getLogger("pipeline")
    logger.debug("Starting pipeline with config: %s", config)

    adapter_factories = build_factories(config["adapters"], ADAPTER_REGISTRY)
    filter_factories = build_factories(config["filters"], FILTER_REGISTRY)

    pipeline = Pipeline(
        datasets=[f() for f in adapter_factories],
        filter_factories=filter_factories,
        sinks=None,
        manifest_db_path=config["pipeline"]["manifest_db"],
        checkpoint_db_path=config["pipeline"]["checkpoint_db"],
        num_workers=config["pipeline"]["num_workers"],
        batch_size=config["pipeline"]["batch_size"],
    )

    pipeline.scan()

    time.sleep(10)
    os._exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <config.json>", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])
