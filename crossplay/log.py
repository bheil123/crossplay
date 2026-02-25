"""
Structured logging for Crossplay engine.

Usage in modules:
    from .log import get_logger
    logger = get_logger(__name__)
    logger.debug("Move generation took %dms", elapsed_ms)
    logger.warning("Score mismatch: expected %d, got %d", expected, actual)

User-facing output stays as print(). Logging is for diagnostics only.
Set CROSSPLAY_LOG_LEVEL=DEBUG env var or call configure() to change level.
"""

import logging
import os
import sys

_configured = False

def configure(level: str = None):
    """Configure the crossplay logger hierarchy.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
               Defaults to CROSSPLAY_LOG_LEVEL env var or WARNING.
    """
    global _configured
    if level is None:
        level = os.environ.get('CROSSPLAY_LOG_LEVEL', 'WARNING')

    root_logger = logging.getLogger('crossplay')
    root_logger.setLevel(getattr(logging, level.upper(), logging.WARNING))

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(
            '[%(levelname).1s] %(name)s: %(message)s'
        ))
        root_logger.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name.

    Auto-configures on first call if not already configured.
    """
    if not _configured:
        configure()
    return logging.getLogger(name)
