"""
Image Sampling Utilities

This module provides various sampling methods for image interpolation.
All sampling operations use float32 precision for calculations.
"""

import numpy as np
from numba import jit, cuda
from ..utils.projection_utils import check_cuda_support

# Check for CUDA support
HAS_CUDA = check_cuda_support()

# CPU sampling methods
@jit(nopython=True)
def nearest_neighbor_sampling(img, y, x):
    """
    Sample image using nearest neighbor interpolation
    
    Parameters:
    - img: Source image
    - y: Y coordinate
    - x: X coordinate
    
    Returns:
    - RGB pixel value
    """
    h, w = img.shape[:2]
    
    # Convert to float, clip, then round to nearest integer
    x_int = int(round(min(max(0.0, float(x)), w - 1)))
    y_int = int(round(min(max(0.0, float(y)), h - 1)))
    
    # Return as float32
    return img[y_int, x_int].astype(np.float32)

@jit(nopython=True)
def bilinear_sampling(img, y, x):
    """
    Sample image using bilinear interpolation
    
    Parameters:
    - img: Source image
    - y: Y coordinate
    - x: X coordinate
    
    Returns:
    - RGB pixel value
    """
    h, w = img.shape[:2]
    
    # Convert to float and clip in one step
    x = min(max(0.0, float(x)), w - 1.001)  # Small epsilon to avoid edge overflow
    y = min(max(0.0, float(y)), h - 1.001)
    
    # Get integer and fractional parts
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    
    # Calculate interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Perform bilinear interpolation for each channel
    result = np.zeros(3, dtype=np.float32)
    for c in range(3):
        top = img[y0, x0, c] * (1.0 - wx) + img[y0, x1, c] * wx
        bottom = img[y1, x0, c] * (1.0 - wx) + img[y1, x1, c] * wx
        result[c] = top * (1.0 - wy) + bottom * wy
    
    return result

@jit(nopython=True)
def sample_image(img, y, x, method="bilinear"):
    """
    Sample image at floating point coordinates using specified sampling method
    
    Parameters:
    - img: Input image (will be converted to float32 internally)
    - y, x: Floating point coordinates to sample at
    - method: Sampling method ('nearest' or 'bilinear')
    
    Returns:
    - Sampled pixel value as float32 array
    """
    if method == "bilinear":
        return bilinear_sampling(img, y, x)
    else:  # Default to nearest neighbor
        return nearest_neighbor_sampling(img, y, x)

# GPU sampling methods (only defined if CUDA is available)
if HAS_CUDA:
    @cuda.jit(device=True)
    def nearest_neighbor_sampling_gpu(img, y, x, w, h):
        """
        Device function for nearest neighbor sampling on GPU
        
        Parameters:
        - img: Source image
        - y, x: Coordinates
        - w, h: Image dimensions
        
        Returns:
        - RGB pixel values as tuple
        """
        # Round and clip coordinates
        px = int(round(min(max(0, x), w-1)))
        py = int(round(min(max(0, y), h-1)))
        
        # Return pixel values as tuple (CUDA can't return arrays from device functions)
        return img[py, px, 0], img[py, px, 1], img[py, px, 2]
    
    @cuda.jit(device=True)
    def bilinear_sampling_gpu(img, y, x, w, h):
        """
        Device function for bilinear sampling on GPU
        
        Parameters:
        - img: Source image
        - y, x: Coordinates
        - w, h: Image dimensions
        
        Returns:
        - RGB pixel values as tuple
        """
        # Ensure coordinates are within bounds
        x = min(max(0, x), w - 1.001)
        y = min(max(0, y), h - 1.001)
        
        # Get integer and fractional parts
        x0 = int(x)
        y0 = int(y)
        x1 = min(x0 + 1, w - 1)
        y1 = min(y0 + 1, h - 1)
        
        # Calculate interpolation weights
        wx = x - x0
        wy = y - y0
        
        # Results for each channel as float32
        r = 0.0
        g = 0.0
        b = 0.0
        
        # Calculate interpolation for each channel
        top_r = img[y0, x0, 0] * (1 - wx) + img[y0, x1, 0] * wx
        bottom_r = img[y1, x0, 0] * (1 - wx) + img[y1, x1, 0] * wx
        r = top_r * (1 - wy) + bottom_r * wy
        
        top_g = img[y0, x0, 1] * (1 - wx) + img[y0, x1, 1] * wx
        bottom_g = img[y1, x0, 1] * (1 - wx) + img[y1, x1, 1] * wx
        g = top_g * (1 - wy) + bottom_g * wy
        
        top_b = img[y0, x0, 2] * (1 - wx) + img[y0, x1, 2] * wx
        bottom_b = img[y1, x0, 2] * (1 - wx) + img[y1, x1, 2] * wx
        b = top_b * (1 - wy) + bottom_b * wy
        
        return r, g, b
