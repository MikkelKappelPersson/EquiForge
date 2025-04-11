"""
EquiForge - A toolkit for equirectangular image processing and conversion.

This package provides tools for converting between different image projection types,
particularly focusing on equirectangular projections.
"""
from equiforge.converters.pers2equi import pers2equi
from equiforge.converters.equi2pers import equi2pers
from equiforge.utils.logging_utils import set_package_log_level, reset_loggers
import logging

from importlib.metadata import version
__version__ = version("equiforge")


__all__ = ['pers2equi', 'equi2pers', 'set_package_log_level', 'reset_loggers']

# Clear any existing handlers and set up a null handler for the package's root logger
root_logger = logging.getLogger('equiforge')
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(logging.NullHandler())
root_logger.propagate = False
