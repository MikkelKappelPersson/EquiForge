"""
Logging utilities for EquiForge.

This module provides consistent logging configuration across the package.
"""

import logging
import sys
from typing import Optional, Union
import io

# Default log format
DEFAULT_LOG_FORMAT = "%(levelname)s: %(message)s"  # Simplified format for notebooks

# Define a custom SILENT level higher than CRITICAL
SILENT = 60  # Standard levels: DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
logging.addLevelName(SILENT, "SILENT")

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

def set_package_log_level(level: Union[int, str, None], show_logs: bool = True, show_initial_message: bool = True) -> None:
    """
    Set the logging level for all EquiForge loggers throughout the application.
    
    Parameters:
    - level: Logging level (e.g., logging.DEBUG, logging.INFO, or "SILENT")
    - show_logs: Whether to ensure logs are displayed on console
    - show_initial_message: Whether to show the "Logging configured" message
    
    This function affects ALL loggers in the equiforge package hierarchy,
    ensuring consistent behavior across different modules.
    """
    # Parse the level if it's a string
    parsed_level = parse_log_level(level)
    
    # Get the root logger for the package
    equiforge_logger = logging.getLogger('equiforge')
    equiforge_logger.setLevel(parsed_level)
    
    # Ensure there's at least one handler that outputs to console for the root logger
    if show_logs and not any(isinstance(h, logging.StreamHandler) for h in equiforge_logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        equiforge_logger.addHandler(handler)
    
    # Update all existing handlers for the root logger
    for handler in equiforge_logger.handlers:
        handler.setLevel(parsed_level)
    
    # Find and update all other loggers in the equiforge hierarchy
    for name in logging.root.manager.loggerDict:
        if name == 'equiforge' or name.startswith('equiforge.'):
            logger = logging.getLogger(name)
            logger.setLevel(parsed_level)
            
            # Update all handlers for this logger
            for handler in logger.handlers:
                handler.setLevel(parsed_level)
    
    # Log message showing the configured level (only if not silenced and message is requested)
    if parsed_level < SILENT and show_initial_message:
        level_name = {
            10: "DEBUG", 
            20: "INFO", 
            30: "WARNING", 
            40: "ERROR", 
            50: "CRITICAL", 
            SILENT: "SILENT"
        }.get(parsed_level, str(parsed_level))
        
        equiforge_logger.info(f"Logging configured successfully at level: {level_name}")

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

def silence_logger(logger, silence=True):
    """
    Temporarily silence a logger by setting its level to SILENT.
    
    Parameters:
    - logger: The logger to silence
    - silence: True to silence, False to restore previous level
    
    Returns:
    - Previous log level if silencing, None if restoring
    """    
    if silence:
        # Store and set to silent level
        previous_level = logger.level
        logger.setLevel(SILENT)
        return previous_level
    else:
        # Restores to default INFO if no level is provided
        logger.setLevel(logging.INFO)
        return None

def parse_log_level(level: Union[int, str, None]) -> int:
    """
    Parse a log level that could be specified as an integer, string, or None.
    
    Parameters:
    - level: Log level as int, string, or None
    
    Returns:
    - Integer log level
    """
    if level is None:
        return SILENT  # None is treated as SILENT for backward compatibility
    
    if isinstance(level, str):
        level_upper = level.upper()
        if level_upper == "SILENT":
            return SILENT
        elif level_upper == "DEBUG":
            return logging.DEBUG
        elif level_upper == "INFO":
            return logging.INFO
        elif level_upper == "WARNING" or level_upper == "WARN":
            return logging.WARNING
        elif level_upper == "ERROR":
            return logging.ERROR
        elif level_upper == "CRITICAL":
            return logging.CRITICAL
        else:
            # Try to convert to int, or default to INFO
            try:
                return int(level)
            except ValueError:
                return logging.INFO
    
    # Already an integer
    return level

# Function to set log level with string or int
def set_log_level(logger, level):
    """
    Set the log level of a logger, supporting string level names.
    
    Parameters:
    - logger: Logger to modify
    - level: Either a valid logging level (int) or level name (str)
    
    Returns:
    - Original log level before change
    """
    original_level = logger.level
    
    # Parse the log level
    parsed_level = parse_log_level(level)
    
    # Set the level
    logger.setLevel(parsed_level)
    
    return original_level
