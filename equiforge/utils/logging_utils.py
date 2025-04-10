"""
Logging utilities for EquiForge.

This module provides consistent logging configuration across the package.
"""

import logging
import sys
from typing import Optional
import io

# Default log format
DEFAULT_LOG_FORMAT = "%(levelname)s: %(message)s"  # Simplified format for notebooks

def setup_logger(
    name: str, 
    level: int = logging.INFO, 
    log_file: Optional[str] = None, 
    format_str: str = DEFAULT_LOG_FORMAT,
    force_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with the specified name and configuration.
    
    Parameters:
    - name: Logger name (usually __name__ of the calling module)
    - level: Logging level (default: INFO)
    - log_file: Optional file to log to (default: None = console only)
    - format_str: Log message format string
    - force_console: Force adding console output even in notebooks
    
    Returns:
    - Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove any existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Create console handler - use sys.stdout for better notebook visibility
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs if parent loggers exist
    logger.propagate = False
    
    return logger

def set_package_log_level(level: int, show_logs: bool = True) -> None:
    """
    Set the logging level for all EquiForge loggers.
    
    Parameters:
    - level: Logging level (e.g., logging.DEBUG, logging.INFO)
    - show_logs: Whether to ensure logs are displayed on console
    """
    # Get the root logger for the package
    equiforge_logger = logging.getLogger('equiforge')
    equiforge_logger.setLevel(level)
    
    # Ensure there's at least one handler that outputs to console
    if show_logs and not any(isinstance(h, logging.StreamHandler) for h in equiforge_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        equiforge_logger.addHandler(handler)
    
    # Update all existing handlers
    for handler in equiforge_logger.handlers:
        handler.setLevel(level)
    
    # Demonstrate that logging is working
    equiforge_logger.info("Logging configured successfully at level: %s", 
                         {10: "DEBUG", 20: "INFO", 30: "WARNING", 40: "ERROR", 50: "CRITICAL"}.get(level, str(level)))

# Function to create a string buffer logger for capturing logs in notebooks
def create_string_logger() -> tuple:
    """
    Create a logger that outputs to a string buffer, useful for notebooks.
    
    Returns:
    - Tuple of (log_capture, log_handler)
    """
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    
    # Add the handler to the root equiforge logger
    logger = logging.getLogger('equiforge')
    logger.addHandler(handler)
    
    return log_capture, handler

def reset_loggers():
    """
    Reset all equiforge loggers by removing handlers.
    Useful for notebooks where cells may be re-run multiple times.
    """
    # Get all loggers
    for name in logging.root.manager.loggerDict:
        # Only reset equiforge loggers
        if name.startswith('equiforge'):
            logger = logging.getLogger(name)
            # Remove all handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            # Add a single NullHandler to prevent "no handler" warnings
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())
    
    # Re-initialize the root package logger
    root_logger = logging.getLogger('equiforge')
    if not root_logger.handlers:
        root_logger.addHandler(logging.NullHandler())
    root_logger.propagate = False
