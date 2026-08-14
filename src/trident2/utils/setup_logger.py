"""
Sets up a simple loguru logger configuration with stdout capture.

"""

import sys
import os
from datetime import datetime
from pathlib import Path
from inspect import stack
from loguru import logger
import psutil


class PrintCapture:
    """Captures print statements and routes them to the logger."""

    def __init__(self, log_level="INFO"):
        self.log_level = log_level

    def write(self, message: str) -> None:
        """Intercept write calls and log them."""
        if message and message.strip():  # Skip empty lines
            logger.log(self.log_level, message.rstrip())

    def flush(self) -> None:
        """No-op flush method for file-like interface."""
        pass


def setup_logger(log_file=None, level="INFO"):
    """
    Configure loguru logger with stderr output and file output.
    Also redirects stdout to capture print statements.

    By default, creates a log file in a 'logs' directory in the caller's directory
    with filename format: run_log_{YYYYMMDD_HHMMSS}.log

    Args:
        log_file: Optional path to write logs to. If None, auto-generates in logs/ directory
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # If log_file not provided, generate default path
    if log_file is None:
        # Get the caller's file path
        caller_frame = stack()[1]
        caller_file = caller_frame.filename
        caller_dir = Path(caller_file).parent

        # Create logs directory
        logs_dir = caller_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"run_log_{timestamp}.log"
        print(f"Logging to file: {log_file}")
    else:
        # Ensure parent directory exists for provided log_file
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_file),
        format="{time} | {level} | {message}",
        colorize=False,
        mode="w",
        level=level,
    )

    # Redirect stdout to capture print statements
    sys.stdout = PrintCapture(log_level=level)

    return logger


def log_memory_usage(logger: logger, stage: str = "") -> None:
    """
    Log current memory usage.

    Args:
        logger: Logger instance to use.
        stage: Optional description of current processing stage.

    Example:
        >>> logger = setup_logging("./logs")
        >>> log_memory_usage(logger, "after loading data")
        DEBUG: Memory usage after loading data: 2.35 GB
    """
    process = psutil.Process()
    memory_gb = process.memory_info().rss / 1024**3
    logger.debug(f"Memory usage {stage}: {memory_gb:.2f} GB")


def get_memory_usage() -> float:
    """
    Get current memory usage in GB.

    Returns:
        Current memory usage in gigabytes.

    Example:
        >>> mem = get_memory_usage()
        >>> print(f"Using {mem:.2f} GB of memory")
    """
    process = psutil.Process()
    return process.memory_info().rss / 1024**3