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
def nearest_neighbor_sampling(img, y, x, wrap_x=False):
    """
    Sample image using nearest neighbor interpolation

    Parameters:
    - img: Source image
    - y: Y coordinate (latitude; clamped, never wrapped)
    - x: X coordinate (longitude; wrapped periodically iff wrap_x=True)
    - wrap_x: If True, wrap x periodically (for equirectangular longitude seam).
      If False (default), clamp x. y is always clamped.

    Returns:
    - RGB pixel value
    """
    h, w = img.shape[:2]

    # Latitude is always clamped, never wrapped (poles do not wrap).
    y_f = min(max(0.0, float(y)), float(h - 1))
    y_int = int(round(y_f))

    if wrap_x:
        # Periodic wrap into [0, w). Uses floor division so negatives wrap correctly.
        x_f = float(x)
        w_f = float(w)
        x_wrapped = x_f - np.floor(x_f / w_f) * w_f
        # Guard against floating-point edge where x_wrapped == w.
        if x_wrapped >= w_f:
            x_wrapped = 0.0
        if x_wrapped < 0.0:
            x_wrapped = 0.0
        x_int = int(round(x_wrapped)) % w
    else:
        # Convert to float, clip, then round to nearest integer
        x_int = int(round(min(max(0.0, float(x)), float(w - 1))))

    # Return as float32
    return img[y_int, x_int].astype(np.float32)

@jit(nopython=True)
def bilinear_sampling(img, y, x, wrap_x=False):
    """
    Sample image using bilinear interpolation

    Parameters:
    - img: Source image
    - y: Y coordinate (latitude; always clamped, never wrapped)
    - x: X coordinate (longitude; wrapped periodically iff wrap_x=True)
    - wrap_x: If True, wrap x across the left/right edge (equirectangular
      longitude seam: interpolate between last and first column).
      If False (default), clamp x (for perspective images where x does not wrap).

    Returns:
    - RGB pixel value
    """
    h, w = img.shape[:2]

    # Latitude is always clamped, never wrapped. Guard upper bound for h == 1.
    y_upper = float(h) - 1.001
    if y_upper < 0.0:
        y_upper = 0.0
    y = min(max(0.0, float(y)), y_upper)

    # Get integer and fractional parts for y (clamped).
    y0 = int(np.floor(y))
    y1 = y0 + 1
    if y1 >= h:
        y1 = h - 1
    wy = y - float(y0)

    if wrap_x:
        # Periodic wrap of longitude into [0, w).
        x_f = float(x)
        w_f = float(w)
        x_wrapped = x_f - np.floor(x_f / w_f) * w_f
        if x_wrapped >= w_f:
            x_wrapped = 0.0
        if x_wrapped < 0.0:
            x_wrapped = 0.0
        x0 = int(np.floor(x_wrapped))
        # Guard against fp edge (x_wrapped == w).
        if x0 >= w:
            x0 = 0
        if x0 < 0:
            x0 = 0
        x1 = x0 + 1
        if x1 >= w:
            x1 = 0
        wx = x_wrapped - float(x0)
    else:
        # Convert to float and clip in one step (guard for w == 1).
        x_upper = float(w) - 1.001  # Small epsilon to avoid edge overflow
        if x_upper < 0.0:
            x_upper = 0.0
        x = min(max(0.0, float(x)), x_upper)  # Small epsilon to avoid edge overflow
        # Get integer and fractional parts
        x0 = int(np.floor(x))
        x1 = x0 + 1
        if x1 >= w:
            x1 = w - 1
        # Calculate interpolation weights
        wx = x - float(x0)

    # Perform bilinear interpolation for each channel
    result = np.zeros(3, dtype=np.float32)
    for c in range(3):
        top = img[y0, x0, c] * (1.0 - wx) + img[y0, x1, c] * wx
        bottom = img[y1, x0, c] * (1.0 - wx) + img[y1, x1, c] * wx
        result[c] = top * (1.0 - wy) + bottom * wy

    return result

@jit(nopython=True)
def sample_image(img, y, x, method="bilinear", wrap_x=False):
    """
    Sample image at floating point coordinates using specified sampling method

    Parameters:
    - img: Input image (will be converted to float32 internally)
    - y, x: Floating point coordinates to sample at
    - method: Sampling method ('nearest' or 'bilinear')
    - wrap_x: If True, wrap x periodically (equirectangular longitude).
      Keep False (default) for perspective images where x does not wrap.

    Returns:
    - Sampled pixel value as float32 array
    """
    if method == "bilinear":
        return bilinear_sampling(img, y, x, wrap_x)
    else:  # Default to nearest neighbor
        return nearest_neighbor_sampling(img, y, x, wrap_x)

# GPU sampling methods (only defined if CUDA is available)
if HAS_CUDA:
    @cuda.jit(device=True)
    def nearest_neighbor_sampling_gpu(img, y, x, w, h, wrap_x=False):
        """
        Device function for nearest neighbor sampling on GPU

        Parameters:
        - img: Source image
        - y, x: Coordinates (y clamped; x wrapped iff wrap_x)
        - w, h: Image dimensions
        - wrap_x: If True, wrap x periodically (equirect seam). Default False.

        Returns:
        - RGB pixel values as tuple
        """
        # Latitude always clamped.
        if y < 0:
            y = 0
        if y > h - 1:
            y = h - 1
        py = int(round(y))
        if py < 0:
            py = 0
        if py >= h:
            py = h - 1

        if wrap_x:
            # Wrap longitude into [0, w).
            x_wrapped = x - np.floor(x / float(w)) * float(w)
            if x_wrapped >= float(w):
                x_wrapped = 0.0
            if x_wrapped < 0.0:
                x_wrapped = 0.0
            px = int(round(x_wrapped)) % w
        else:
            # Round and clip coordinates
            if x < 0:
                x = 0
            if x > w - 1:
                x = w - 1
            px = int(round(x))

        # Return pixel values as tuple (CUDA can't return arrays from device functions)
        return img[py, px, 0], img[py, px, 1], img[py, px, 2]

    @cuda.jit(device=True)
    def bilinear_sampling_gpu(img, y, x, w, h, wrap_x=False):
        """
        Device function for bilinear sampling on GPU

        Parameters:
        - img: Source image
        - y, x: Coordinates (y clamped; x wrapped iff wrap_x)
        - w, h: Image dimensions
        - wrap_x: If True, wrap x across the seam (interpolate last<->first
          column). Keep False for perspective images. Default False.

        Returns:
        - RGB pixel values as tuple
        """
        # Latitude always clamped, never wrapped.
        if y < 0:
            y = 0
        if y > h - 1.001:
            y = h - 1.001
        y0 = int(y)
        y1 = y0 + 1
        if y1 >= h:
            y1 = h - 1
        wy = y - y0

        if wrap_x:
            x_wrapped = x - np.floor(x / float(w)) * float(w)
            if x_wrapped >= float(w):
                x_wrapped = 0.0
            if x_wrapped < 0.0:
                x_wrapped = 0.0
            x0 = int(np.floor(x_wrapped))
            if x0 >= w:
                x0 = 0
            if x0 < 0:
                x0 = 0
            x1 = x0 + 1
            if x1 >= w:
                x1 = 0
            wx = x_wrapped - x0
        else:
            # Ensure coordinates are within bounds
            if x < 0:
                x = 0
            if x > w - 1.001:
                x = w - 1.001
            # Get integer and fractional parts
            x0 = int(x)
            x1 = x0 + 1
            if x1 >= w:
                x1 = w - 1
            # Calculate interpolation weights
            wx = x - x0
        
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
