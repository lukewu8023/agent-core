# utils/logger.py

import logging
import os


def get_logger(name="agent-core", log_level: str = None) -> logging.Logger:
    """
    Retrieve a logger with the specified name.
    If log_level is provided, override the default or global environment setting.
    Otherwise, use Config.DEFAULT_LOG_LEVEL.
    """
    logger = logging.getLogger(name)

    # If the logger has no handlers, add one (to avoid duplicated logs in multi-import environments).
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # If a log_level is specified, use it; otherwise use the framework default
    effective_level = log_level.upper() if log_level else os.getenv("AGENT_CORE_LOG_LEVEL")
    if not effective_level:
        effective_level = "WARNING"
    logger.setLevel(effective_level)

    return logger
