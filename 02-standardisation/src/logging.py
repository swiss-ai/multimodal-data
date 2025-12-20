import logging
import sys
from pathlib import Path

LOG_NAME = "pipeline"
LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int, log_file: str | None) -> logging.Logger:
    """
    Configure logging for the data pipeline.

    Args:
        level:    log level (e.g., logging.INFO)
        log_file: path to log file (if None, no file logging)
    """

    logger = logging.getLogger(LOG_NAME)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, LOG_DATEFMT)

    # stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # file handler
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # reduce noise from other libraries
    for lib in ["urllib3", "datasets", "huggingface_hub", "PIL"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOG_NAME}.{name}")
