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
try:
    __version__ = version("equiforge")
except:
    __version__ = "0.1.0"  # fallback version

__all__ = ['pers2equi', 'equi2pers', 'set_package_log_level', 'reset_loggers']

# Clear any existing handlers and set up a null handler for the package's root logger
root_logger = logging.getLogger('equiforge')
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
root_logger.addHandler(logging.NullHandler())
root_logger.propagate = False

# Set default log level based on version
# If it's a development version (contains '+' or '.dev'), use INFO level
# For released versions on main branch, use WARNING level
if '+' in __version__ or '.dev' in __version__:
    # Development version
    default_log_level = logging.INFO
else:
    # Release version
    default_log_level = logging.WARNING

# Use silent level during initial import to avoid logs unless users specifically enable them
set_package_log_level(default_log_level, show_initial_message=False)
