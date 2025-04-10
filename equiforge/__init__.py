"""
EquiForge - A toolkit for equirectangular image processing and conversion.

This package provides tools for converting between different image projection types,
particularly focusing on equirectangular projections.
"""
from equiforge.converters.pers2equi import pers2equi
from equiforge.utils.logging_utils import set_package_log_level, reset_loggers
import logging

__version__ = "0.1.0"
__all__ = ['pers2equi', 'set_package_log_level', 'reset_loggers']

# Clear any existing handlers and set up a null handler for the package's root logger
root_logger = logging.getLogger('equiforge')
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(logging.NullHandler())
root_logger.propagate = False
