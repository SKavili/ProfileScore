"""
Logging configuration for ProfileScore application.
"""

import sys
import os
from loguru import logger
from typing import Optional


def setup_logger(log_level: Optional[str] = None) -> None:
    """
    Configure loguru logger with custom formatting and handlers.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default handler
    logger.remove()
    
    # Get log level from environment or use default
    level = log_level or os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Console handler with color formatting
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File handler for persistent logs
    logger.add(
        "logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )
    
    # Error file handler
    logger.add(
        "logs/error.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip"
    )


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Initialize logger on module import
setup_logger() 