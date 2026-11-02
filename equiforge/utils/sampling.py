"""
Image Sampling Utilities (torch-only backend).

Single-coordinate helpers preserved for backward compatibility and unit tests.
Vectorized converters use torch advanced indexing directly; these numpy helpers
implement the same semantics: x wraps periodically iff wrap_x=True (equirect
seam), y is always clamped (poles never wrap).
"""

import numpy as np


def nearest_neighbor_sampling(img, y, x, wrap_x=False):
    """
    Sample image using nearest neighbor interpolation.

    Parameters:
    - img: Source image (H,W,3)
    - y: Y coordinate (latitude; clamped, never wrapped)
    - x: X coordinate (longitude; wrapped periodically iff wrap_x=True)
    - wrap_x: If True, wrap x periodically. Default False.

    Returns:
    - RGB pixel value as float32 array
    """
    h, w = img.shape[:2]
    y_f = min(max(0.0, float(y)), float(h - 1))
    y_int = int(round(y_f))
    if wrap_x:
        x_f = float(x)
        w_f = float(w)
        x_wrapped = x_f - np.floor(x_f / w_f) * w_f
        if x_wrapped >= w_f:
            x_wrapped = 0.0
        if x_wrapped < 0.0:
            x_wrapped = 0.0
        x_int = int(round(x_wrapped)) % w
    else:
        x_int = int(round(min(max(0.0, float(x)), float(w - 1))))
    return np.asarray(img[y_int, x_int], dtype=np.float32)


def bilinear_sampling(img, y, x, wrap_x=False):
    """
    Sample image using bilinear interpolation.

    Parameters:
    - img: Source image (H,W,3)
    - y: Y coordinate (latitude; always clamped)
    - x: X coordinate (wrapped iff wrap_x=True)
    - wrap_x: If True, wrap x across the seam (interpolate last<->first).

    Returns:
    - RGB pixel value as float32 array
    """
    h, w = img.shape[:2]
    y_upper = float(h) - 1.001
    if y_upper < 0.0:
        y_upper = 0.0
    y = min(max(0.0, float(y)), y_upper)
    y0 = int(np.floor(y))
    y1 = y0 + 1
    if y1 >= h:
        y1 = h - 1
    wy = y - float(y0)

    if wrap_x:
        x_f = float(x)
        w_f = float(w)
        x_wrapped = x_f - np.floor(x_f / w_f) * w_f
        if x_wrapped >= w_f:
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
        wx = x_wrapped - float(x0)
    else:
        x_upper = float(w) - 1.001
        if x_upper < 0.0:
            x_upper = 0.0
        x = min(max(0.0, float(x)), x_upper)
        x0 = int(np.floor(x))
        x1 = x0 + 1
        if x1 >= w:
            x1 = w - 1
        wx = x - float(x0)

    img_f = np.asarray(img, dtype=np.float32)
    result = np.zeros(3, dtype=np.float32)
    for c in range(3):
        top = img_f[y0, x0, c] * (1.0 - wx) + img_f[y0, x1, c] * wx
        bottom = img_f[y1, x0, c] * (1.0 - wx) + img_f[y1, x1, c] * wx
        result[c] = top * (1.0 - wy) + bottom * wy
    return result


def sample_image(img, y, x, method="bilinear", wrap_x=False):
    """
    Sample image at floating point coordinates.

    Parameters:
    - img: Input image
    - y, x: Floating point coordinates
    - method: 'nearest' or 'bilinear'
    - wrap_x: If True, wrap x periodically. Default False.

    Returns:
    - Sampled pixel value as float32 array
    """
    if method == "bilinear":
        return bilinear_sampling(img, y, x, wrap_x)
    return nearest_neighbor_sampling(img, y, x, wrap_x)
