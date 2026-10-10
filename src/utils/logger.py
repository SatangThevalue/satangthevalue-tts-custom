import logging
import os
from pathlib import Path
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configures clean, concise console logger (default: INFO) while writing
    detailed DEBUG logs to persistent Google Drive file.
    Eliminates console flooding on Google Colab.
    """
    env_level = os.getenv("LOG_LEVEL", level).upper()
    console_level_name = os.getenv("CONSOLE_LOG_LEVEL", env_level).upper()

    console_level = getattr(logging, console_level_name, logging.INFO)
    file_level = logging.DEBUG  # File always captures complete debug history

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not logger.handlers:
        # 1. Clean Console Stream Handler (INFO by default to avoid flooding)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. Detailed File Handler for Google Drive persistence (Full DEBUG)
        dest_log_file = log_file or os.getenv(
            "LOG_FILE", "/content/drive/MyDrive/tts-project/logs/pipeline_debug.log"
        )
        try:
            if dest_log_file and ("/content/drive" in dest_log_file or "logs" in dest_log_file):
                log_path = Path(dest_log_file)
                if log_path.parent.exists() or "/content/drive" not in dest_log_file:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
                    file_handler.setLevel(file_level)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
        except Exception:
            pass

    return logger
