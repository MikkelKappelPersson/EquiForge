"""
Perspective to Equirectangular Converter (torch-only backend).

Vectorized torch implementation for CPU and CUDA. No Numba, no Pool.
Perspective source never wraps (wrap_x=False); out-of-view rays stay black.
"""

import numpy as np
from PIL import Image
import logging

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False
    HAS_CUDA = False

from ..utils.projection_utils import create_rotation_matrix, calculate_focal_length, timer
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def _require_torch():
    if not HAS_TORCH:
        raise ImportError("torch is required (pip install torch)")


def _resolve_device(use_gpu=True, device=None):
    _require_torch()
    if device is not None:
        return torch.device(device)
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_tensor(img, device):
    if isinstance(img, torch.Tensor):
        t = img.to(device=device, dtype=torch.float32)
        if t.ndim != 3 or t.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) tensor, got {tuple(t.shape)}")
        return t
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Input image must have shape (height, width, 3), got {arr.shape}")
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def _render_chunk(img_t, xs, output_height, output_width, cx, cy, f_h, f_v, R_t, sampling_method):
    """Render output columns xs -> (H_out, len(xs), 3). Unseen rays are black."""
    dev = img_t.device
    h, w = img_t.shape[0], img_t.shape[1]
    ys = torch.arange(output_height, device=dev, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs.to(device=dev, dtype=torch.float32), indexing="ij")
    pi = float(np.pi)
    phi = pi * grid_y / output_height - pi / 2
    theta = 2 * pi * grid_x / output_width - pi

    vec_x = torch.cos(phi) * torch.sin(theta)
    vec_y = torch.sin(phi)
    vec_z = torch.cos(phi) * torch.cos(theta)
    rx = R_t[0, 0] * vec_x + R_t[0, 1] * vec_y + R_t[0, 2] * vec_z
    ry = R_t[1, 0] * vec_x + R_t[1, 1] * vec_y + R_t[1, 2] * vec_z
    rz = R_t[2, 0] * vec_x + R_t[2, 1] * vec_y + R_t[2, 2] * vec_z

    valid_front = rz > 0
    # avoid div-by-zero behind camera; values masked out later
    rz_safe = torch.where(valid_front, rz, torch.ones_like(rz))
    px = f_h * rx / rz_safe + cx
    py = f_v * ry / rz_safe + cy
    valid = valid_front & (px >= 0) & (px < w) & (py >= 0) & (py < h)

    if sampling_method == "nearest":
        y_c = torch.clamp(py, 0.0, float(h - 1))
        x_c = torch.clamp(px, 0.0, float(w - 1))
        y_int = torch.round(y_c).long().clamp(0, h - 1)
        x_int = torch.round(x_c).long().clamp(0, w - 1)
        sampled = img_t[y_int, x_int]
    else:
        y_upper = float(h) - 1.001
        if y_upper < 0.0:
            y_upper = 0.0
        x_upper = float(w) - 1.001
        if x_upper < 0.0:
            x_upper = 0.0
        y_c = torch.clamp(py, 0.0, y_upper)
        x_c = torch.clamp(px, 0.0, x_upper)
        y0 = torch.floor(y_c).long().clamp(0, h - 1)
        x0 = torch.floor(x_c).long().clamp(0, w - 1)
        y1 = torch.clamp(y0 + 1, max=h - 1)
        x1 = torch.clamp(x0 + 1, max=w - 1)
        wy = (y_c - y0.float()).unsqueeze(-1)
        wx = (x_c - x0.float()).unsqueeze(-1)
        top = img_t[y0, x0] * (1.0 - wx) + img_t[y0, x1] * wx
        bottom = img_t[y1, x0] * (1.0 - wx) + img_t[y1, x1] * wx
        sampled = top * (1.0 - wy) + bottom * wy

    out = torch.where(valid.unsqueeze(-1), sampled, torch.zeros_like(sampled))
    return out


def _prepare(img, output_height, fov_x, yaw, pitch, roll, device):
    img_t = _to_tensor(img, device)
    h, w = img_t.shape[0], img_t.shape[1]
    output_width = output_height * 2
    fov_x = max(0.1, min(179.9, fov_x))
    cx, cy = w // 2, h // 2
    fov_x_rad, yaw_rad, pitch_rad, roll_rad = map(np.radians, [fov_x, yaw, pitch, roll])
    f_h, f_v = calculate_focal_length(w, h, fov_x_rad)
    R = create_rotation_matrix(yaw_rad, pitch_rad, roll_rad).astype(np.float32)
    R_t = torch.as_tensor(R, device=device, dtype=torch.float32)
    return img_t, output_width, cx, cy, float(f_h), float(f_v), R_t


def process_chunk(args):
    """Backward-compatible chunk helper (single-process, torch)."""
    if len(args) == 7:
        img, x_start, x_end, output_height, params, sampling_method, output_dtype = args
    elif len(args) == 6:
        img, x_start, x_end, output_height, params, sampling_method = args
        output_dtype = "uint8"
    else:
        raise ValueError(f"process_chunk expects 6- or 7-tuple, got len {len(args)}")
    if output_dtype not in ("uint8", "float32"):
        raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
    fov_x, yaw, pitch, roll = params[:4]
    dev = torch.device("cpu")
    img_t, output_width, cx, cy, f_h, f_v, R_t = _prepare(
        img, output_height, fov_x, yaw, pitch, roll, dev)
    xs = torch.arange(x_start, x_end, dtype=torch.float32)
    chunk = _render_chunk(img_t, xs, output_height, output_width, cx, cy, f_h, f_v, R_t, sampling_method)
    if output_dtype == "float32":
        return chunk.detach().cpu().numpy().astype(np.float32, copy=False), x_start, x_end
    return torch.clamp(chunk, 0.0, 255.0).to(torch.uint8).cpu().numpy(), x_start, x_end


def pers2equi_cpu(img, output_height,
                  fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0,
                  sampling_method="bilinear", output_dtype="uint8",
                  device=None, as_tensor=False):
    """CPU conversion (torch, device=cpu)."""
    if output_dtype not in ("uint8", "float32"):
        raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
    dev = torch.device("cpu") if device is None else torch.device(device)
    img_t, output_width, cx, cy, f_h, f_v, R_t = _prepare(
        img, output_height, fov_x, yaw, pitch, roll, dev)
    xs = torch.arange(output_width, dtype=torch.float32)
    out = _render_chunk(img_t, xs, output_height, output_width, cx, cy, f_h, f_v, R_t, sampling_method)
    if output_dtype == "uint8":
        out = torch.clamp(out, 0.0, 255.0).to(torch.uint8)
    if as_tensor:
        return out
    arr = out.detach().cpu().numpy()
    return arr.astype(np.float32, copy=False) if output_dtype == "float32" else arr


@timer
def pers2equi_gpu(img, output_height,
                  fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0,
                  sampling_method="bilinear", output_dtype="uint8",
                  device=None, as_tensor=False):
    """GPU conversion (torch CUDA). Raises if CUDA unavailable."""
    _require_torch()
    if output_dtype not in ("uint8", "float32"):
        raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
    dev = torch.device(device) if device is not None else torch.device("cuda")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    img_t, output_width, cx, cy, f_h, f_v, R_t = _prepare(
        img, output_height, fov_x, yaw, pitch, roll, dev)
    xs = torch.arange(output_width, dtype=torch.float32)
    out = _render_chunk(img_t, xs, output_height, output_width, cx, cy, f_h, f_v, R_t, sampling_method)
    if output_dtype == "uint8":
        out = torch.clamp(out, 0.0, 255.0).to(torch.uint8)
    if as_tensor:
        return out
    arr = out.detach().cpu().numpy()
    return arr.astype(np.float32, copy=False) if output_dtype == "float32" else arr


def pers2equi(img, output_height,
              fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0,
              use_gpu=True, sampling_method="bilinear", output_dtype="uint8",
              device=None, as_tensor=False):
    """
    Convert perspective image to equirectangular projection (torch backend).

    Parameters:
    - img: Input perspective image (numpy array, torch tensor, or file path)
    - output_height: Height of output (width is 2x height)
    - fov_x, yaw, pitch, roll: View parameters in degrees
    - use_gpu: Use CUDA if available (ignored if device is given)
    - sampling_method: "nearest" or "bilinear" (default)
    - output_dtype: "uint8" (default, clip [0,255]) or "float32" (raw)
    - device: Optional torch device. Overrides use_gpu.
    - as_tensor: If True return torch tensor, else numpy.

    Returns:
    - Equirectangular image as numpy array (default) or torch tensor.
    """
    if output_dtype not in ("uint8", "float32"):
        raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
    logger.info(f"pers2equi called with output_height={output_height}, fov_x={fov_x}, use_gpu={use_gpu}")

    if isinstance(img, str):
        logger.info(f"Loading image from path: {img}")
        img = np.array(Image.open(img))

    if img is None:
        raise ValueError("Input image cannot be None")

    if isinstance(img, np.ndarray):
        logger.info(f"Input image shape: {img.shape}, dtype: {img.dtype}")
        if len(img.shape) != 3 or img.shape[2] != 3:
            raise ValueError(f"Input image must have exactly 3 color channels, got shape {img.shape}")
    elif torch is not None and isinstance(img, torch.Tensor):
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Input image must have exactly 3 color channels, got shape {tuple(img.shape)}")
    else:
        arr = np.asarray(img)
        if len(arr.shape) != 3 or arr.shape[2] != 3:
            raise ValueError(f"Input image must have exactly 3 color channels, got shape {arr.shape}")
        img = arr

    fov_x = max(0.1, min(179.9, fov_x))
    logger.info(f"Processing with parameters: FOV={fov_x}°, yaw={yaw}°, pitch={pitch}°, roll={roll}°")

    dev = _resolve_device(use_gpu=use_gpu, device=device)
    logger.info(f"Using device {dev}")
    img_t, output_width, cx, cy, f_h, f_v, R_t = _prepare(
        img, output_height, fov_x, yaw, pitch, roll, dev)
    xs = torch.arange(output_width, dtype=torch.float32)
    out = _render_chunk(img_t, xs, output_height, output_width, cx, cy, f_h, f_v, R_t, sampling_method)
    if output_dtype == "uint8":
        out = torch.clamp(out, 0.0, 255.0).to(torch.uint8)
    if as_tensor:
        return out
    arr = out.detach().cpu().numpy()
    result = arr.astype(np.float32, copy=False) if output_dtype == "float32" else arr
    if np.sum(arr) == 0:
        logger.warning("Output image is all zeros - this may indicate a problem with input parameters")
    else:
        logger.info("Processing completed successfully")
    return result
