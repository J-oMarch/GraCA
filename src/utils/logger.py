import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Create a logger with file and console handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(f"{log_dir}/{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger
