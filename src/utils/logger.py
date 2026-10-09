import logging
import os
from pathlib import Path
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: str = "DEBUG",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configures unified logger with DEBUG level and detailed formatting.
    Optionally outputs to persistent file on Google Drive (LOG_FILE env var).
    """
    env_level = os.getenv("LOG_LEVEL", level).upper()
    log_level = getattr(logging, env_level, logging.DEBUG)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        # 1. Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler for Google Drive persistence
        dest_log_file = log_file or os.getenv("LOG_FILE", "/content/drive/MyDrive/tts-project/logs/pipeline_debug.log")
        # Only attach file handler if running in Colab or parent directory exists/creatable
        try:
            if dest_log_file and ("/content/drive" in dest_log_file or "logs" in dest_log_file):
                log_path = Path(dest_log_file)
                if log_path.parent.exists() or "/content/drive" not in dest_log_file:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
                    file_handler.setLevel(log_level)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
        except Exception:
            pass  # Fallback to console only if filesystem is restricted

    return logger
