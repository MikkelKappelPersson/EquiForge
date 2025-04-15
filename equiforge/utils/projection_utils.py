"""
Utility functions for image projection transformations.
"""

import numpy as np
import time
import logging
from numba import cuda
from .logging_utils import setup_logger, SILENT

# Set up logger
logger = setup_logger(__name__)

def create_rotation_matrix(yaw, pitch, roll):
    """
    Create a combined rotation matrix from yaw, pitch, and roll angles.
    
    Parameters:
    - yaw: Rotation around vertical axis (left/right) in radians
    - pitch: Rotation around horizontal axis (up/down) in radians
    - roll: Rotation around depth axis (clockwise/counterclockwise) in radians
    
    Returns:
    - Combined rotation matrix
    """
    # Create rotation matrices
    # For yaw: positive = right rotation
    R_yaw = np.array([
        [np.cos(yaw), 0, -np.sin(yaw)],
        [0, 1, 0],
        [np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # For pitch: positive = up rotation
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), np.sin(pitch)],
        [0, -np.sin(pitch), np.cos(pitch)]
    ])
    
    # Roll stays the same
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])
    
    # Change order to apply roll last, after view direction (yaw/pitch) is established
    # This ensures roll only rotates around the viewing axis
    return R_roll @ R_pitch @ R_yaw

def calculate_focal_length(width, height, fov_x_rad):
    """
    Calculate focal lengths based on image dimensions and field of view.
    
    Parameters:
    - width, height: Image dimensions
    - fov_x_rad: Horizontal field of view in radians
    
    Returns:
    - Tuple of (horizontal focal length, vertical focal length)
    """
    # Calculate vertical FOV based on aspect ratio
    aspect_ratio = width / height
    fov_y_rad = fov_x_rad / aspect_ratio
    
    # Calculate focal lengths
    f_h = (width / 2) / np.tan(fov_x_rad / 2)
    f_v = (height / 2) / np.tan(fov_y_rad / 2)
    
    return f_h, f_v

def check_cuda_support(quiet=True):
    """
    Check if CUDA is available and return status.
    
    Parameters:
    - quiet: If True, suppresses log output during initialization
    
    Returns:
    - Boolean indicating CUDA availability
    """
    has_cuda = cuda.is_available()
    if not quiet:
        if has_cuda:
            logger.info("CUDA support found: GPU acceleration available")
        else:
            logger.info("No compatible GPU detected, using CPU acceleration only")
    return has_cuda

def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # Check the package root logger level instead of just this module's logger
        package_logger = logging.getLogger('equiforge')
        if package_logger.level < SILENT:
            logger.info(f"{func.__name__} completed in {elapsed:.2f} seconds.")
        return result
    return wrapper
