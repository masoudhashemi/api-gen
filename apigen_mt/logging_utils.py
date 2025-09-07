from __future__ import annotations
import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    level_name = level or os.getenv("APIGEN_LOG_LEVEL", "INFO")
    lvl = getattr(logging, level_name.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=lvl, format=fmt, datefmt=datefmt)

