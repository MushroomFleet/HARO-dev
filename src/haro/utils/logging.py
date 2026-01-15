"""Structured logging configuration for HARO.

Provides consistent logging across all HARO components using structlog.
Supports both console (human-readable) and file (JSON) output formats.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import structlog
from structlog.typing import Processor


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    ui_mode: bool = False,
) -> None:
    """Configure structlog for HARO.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Path to log file. If None, file logging is disabled.
        console: Whether to output to console.
        max_file_size: Maximum size of log file before rotation.
        backup_count: Number of backup log files to keep.
        ui_mode: If True, suppress console output to avoid cluttering the rich UI.
    """
    # When UI mode is enabled, disable console logging to keep the display clean
    if ui_mode:
        console = False
        # If no log file specified in UI mode, log to a default location
        if not log_file:
            log_file = ".context/logs/haro.log"
    # Convert level string to logging level
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors for structlog
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
    ]

    # Configure structlog
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up handlers for stdlib logging
    handlers: list[logging.Handler] = []

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.dev.ConsoleRenderer(colors=True),
                foreign_pre_chain=shared_processors,
            )
        )
        handlers.append(console_handler)

    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=shared_processors,
            )
        )
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        handler.setLevel(log_level)
        root_logger.addHandler(handler)

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger for the given name.

    Args:
        name: Logger name, typically __name__ of the module.

    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)
