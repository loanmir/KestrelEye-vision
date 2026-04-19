"""
src/utils/logger.py
--------------------
Configures Python logging for the whole KestrelEye project.

Call setup_logging() once at the top of any entry-point script.
All child loggers inside src/ inherit the configuration automatically.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    run_name: str | None = None,
) -> logging.Logger:
    """
    Configure root logger with a console handler and an optional file handler.

    Parameters
    ----------
    level : str
        "DEBUG" | "INFO" | "WARNING" | "ERROR"
    log_dir : str | Path | None
        If provided, also write logs to <log_dir>/<run_name>.log.
    run_name : str | None
        Log file stem. Defaults to "kestreleye_<timestamp>".

    Returns
    -------
    logging.Logger  The configured root logger.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  --  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers.clear()

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File (optional)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stem     = run_name or f"kestreleye_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_file = log_dir / f"{stem}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.info("Log file: %s", log_file)

    # Silence noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return root