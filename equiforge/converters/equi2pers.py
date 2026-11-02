"""
Equirectangular to Perspective Converter (torch-only backend).

Vectorized torch implementation for CPU and CUDA. No Numba, no multiprocessing
Pool. Longitude wraps periodically (seam interpolates last<->first), latitude
clamps (poles never wrap). HDR-safe via output_dtype="float32".
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

# Set up logger
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


def _to_tensor(equi, device):
    if isinstance(equi, torch.Tensor):
        t = equi.to(device=device, dtype=torch.float32)
        if t.ndim != 3 or t.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) tensor, got {tuple(t.shape)}")
        return t
    arr = np.asarray(equi)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Input image must have shape (height, width, 3), got {arr.shape}")
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def _render(equi_t, xs, output_height, cx, cy, f_h, f_v, R_inv_t, sampling_method):
    """Render columns xs (1D global x coords) -> (H_out, len(xs), 3) float32."""
    dev = equi_t.device
    equi_h, equi_w = equi_t.shape[0], equi_t.shape[1]
    ys = torch.arange(output_height, device=dev, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs.to(device=dev, dtype=torch.float32), indexing="ij")

    ndc_x = (grid_x - cx) / f_h
    ndc_y = (grid_y - cy) / f_v
    dir_x = ndc_x
    dir_y = -ndc_y
    dir_z = torch.ones_like(dir_x)
    norm = torch.sqrt(dir_x * dir_x + dir_y * dir_y + dir_z * dir_z)
    dir_x = dir_x / norm
    dir_y = dir_y / norm
    dir_z = dir_z / norm

    world_x = R_inv_t[0, 0] * dir_x + R_inv_t[0, 1] * dir_y + R_inv_t[0, 2] * dir_z
    world_y = R_inv_t[1, 0] * dir_x + R_inv_t[1, 1] * dir_y + R_inv_t[1, 2] * dir_z
    world_z = R_inv_t[2, 0] * dir_x + R_inv_t[2, 1] * dir_y + R_inv_t[2, 2] * dir_z

    theta = torch.atan2(world_x, world_z)
    phi = torch.asin(torch.clamp(world_y, -1.0, 1.0))
    pi = float(np.pi)
    equi_x = (theta + pi) / (2 * pi) * float(equi_w)
    equi_y = (pi / 2 - phi) / pi * float(equi_h)

    if sampling_method == "nearest":
        y_c = torch.clamp(equi_y, 0.0, float(equi_h - 1))
        y_int = torch.round(y_c).long().clamp(0, equi_h - 1)
        w_f = float(equi_w)
        x_wrapped = equi_x - torch.floor(equi_x / w_f) * w_f
        x_wrapped = torch.where(x_wrapped >= w_f, torch.zeros_like(x_wrapped), x_wrapped)
        x_wrapped = torch.where(x_wrapped < 0.0, torch.zeros_like(x_wrapped), x_wrapped)
        x_int = (torch.round(x_wrapped).long() % equi_w)
        return equi_t[y_int, x_int]

    # bilinear, wrap x / clamp y
    y_upper = float(equi_h) - 1.001
    if y_upper < 0.0:
        y_upper = 0.0
    y_c = torch.clamp(equi_y, 0.0, y_upper)
    y0 = torch.floor(y_c).long().clamp(0, equi_h - 1)
    y1 = torch.clamp(y0 + 1, max=equi_h - 1)
    wy = (y_c - y0.float()).unsqueeze(-1)

    w_f = float(equi_w)
    x_wrapped = equi_x - torch.floor(equi_x / w_f) * w_f
    x_wrapped = torch.where(x_wrapped >= w_f, torch.zeros_like(x_wrapped), x_wrapped)
    x_wrapped = torch.where(x_wrapped < 0.0, torch.zeros_like(x_wrapped), x_wrapped)
    x0 = torch.floor(x_wrapped).long().clamp(0, equi_w - 1)
    x1 = (x0 + 1) % equi_w
    wx = (x_wrapped - x0.float()).unsqueeze(-1)

    top = equi_t[y0, x0] * (1.0 - wx) + equi_t[y0, x1] * wx
    bottom = equi_t[y1, x0] * (1.0 - wx) + equi_t[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _prepare(equi, output_width, output_height, fov_x, yaw, pitch, roll, device):
    equi_t = _to_tensor(equi, device)
    equi_h, equi_w = equi_t.shape[0], equi_t.shape[1]
    fov_x = max(0.1, min(179.9, fov_x))
    cx = output_width // 2
    cy = output_height // 2
    fov_x_rad, yaw_rad, pitch_rad, roll_rad = map(np.radians, [fov_x, yaw, pitch, roll])
    f_h, f_v = calculate_focal_length(output_width, output_height, fov_x_rad)
    R = create_rotation_matrix(yaw_rad, pitch_rad, roll_rad)
    R_inv = np.linalg.inv(R).astype(np.float32)
    R_inv_t = torch.as_tensor(R_inv, device=device, dtype=torch.float32)
    return equi_t, cx, cy, float(f_h), float(f_v), R_inv_t


def process_chunk(args):
    """Backward-compatible chunk helper (single-process, torch).

    Accepts legacy 7-tuple (..., sampling_method) or 8-tuple (+ output_dtype).
    Returns (chunk, x_start, x_end) with dtype matching output_dtype.
    """
    if len(args) == 8:
        equi, x_start, x_end, output_width, output_height, params, sampling_method, output_dtype = args
    else:
        equi, x_start, x_end, output_width, output_height, params, sampling_method = args
        output_dtype = "uint8"
    if output_dtype not in ("uint8", "float32"):
        raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
    fov_x, yaw, pitch, roll = params
    dev = torch.device("cpu")
    equi_t, cx, cy, f_h, f_v, R_inv_t = _prepare(
        equi, output_width, output_height, fov_x, yaw, pitch, roll, dev)
    xs = torch.arange(x_start, x_end, dtype=torch.float32)
    chunk = _render(equi_t, xs, output_height, cx, cy, f_h, f_v, R_inv_t, sampling_method)
    if output_dtype == "float32":
        return chunk.detach().cpu().numpy().astype(np.float32, copy=False), x_start, x_end
    return torch.clamp(chunk, 0.0, 255.0).to(torch.uint8).cpu().numpy(), x_start, x_end


def equi2pers_cpu(equi, output_width, output_height,
                  fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0,
                  sampling_method="bilinear", output_dtype="uint8",
                  device=None, as_tensor=False):
    """CPU conversion (torch, device=cpu). See equi2pers for params."""
    if output_dtype not in ("uint8", "float32"):
        raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
    if len(np.asarray(equi).shape) != 3 or np.asarray(equi).shape[2] != 3:
        raise ValueError(f"Input image must have shape (height, width, 3), got {np.asarray(equi).shape}")
    dev = torch.device("cpu") if device is None else torch.device(device)
    equi_t, cx, cy, f_h, f_v, R_inv_t = _prepare(
        equi, output_width, output_height, fov_x, yaw, pitch, roll, dev)
    xs = torch.arange(output_width, dtype=torch.float32)
    out = _render(equi_t, xs, output_height, cx, cy, f_h, f_v, R_inv_t, sampling_method)
    if output_dtype == "uint8":
        out = torch.clamp(out, 0.0, 255.0).to(torch.uint8)
    if as_tensor:
        return out
    arr = out.detach().cpu().numpy()
    return arr.astype(np.float32, copy=False) if output_dtype == "float32" else arr


@timer
def equi2pers_gpu(equi, output_width, output_height,
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
    if len(np.asarray(equi).shape) != 3 and not isinstance(equi, torch.Tensor):
        raise ValueError("Input image must have shape (height, width, 3)")
    equi_t, cx, cy, f_h, f_v, R_inv_t = _prepare(
        equi, output_width, output_height, fov_x, yaw, pitch, roll, dev)
    xs = torch.arange(output_width, dtype=torch.float32)
    out = _render(equi_t, xs, output_height, cx, cy, f_h, f_v, R_inv_t, sampling_method)
    if output_dtype == "uint8":
        out = torch.clamp(out, 0.0, 255.0).to(torch.uint8)
    if as_tensor:
        return out
    arr = out.detach().cpu().numpy()
    return arr.astype(np.float32, copy=False) if output_dtype == "float32" else arr


def equi2pers(img, output_width, output_height,
              fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0,
              use_gpu=True, sampling_method="bilinear", output_dtype="uint8",
              device=None, as_tensor=False):
    """
    Convert equirectangular image to perspective projection (torch backend).

    Parameters:
    - img: Input equirectangular image (numpy array, torch tensor, or file path)
    - output_width, output_height: Output size
    - fov_x, yaw, pitch, roll: View parameters in degrees
    - use_gpu: Use CUDA if available (ignored if device is given)
    - sampling_method: "bilinear" (default) or "nearest"
    - output_dtype: "uint8" (default, clip [0,255]) or "float32" (raw HDR)
    - device: Optional torch device (e.g. "cpu", "cuda"). Overrides use_gpu.
    - as_tensor: If True return torch tensor on device, else numpy.

    Returns:
    - Perspective image as numpy array (default) or torch tensor.
    """
    if output_dtype not in ("uint8", "float32"):
        raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
    logger.info(f"equi2pers called with output_width={output_width}, output_height={output_height}, fov_x={fov_x}, use_gpu={use_gpu}")

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
    equi_t, cx, cy, f_h, f_v, R_inv_t = _prepare(
        img, output_width, output_height, fov_x, yaw, pitch, roll, dev)
    xs = torch.arange(output_width, dtype=torch.float32)
    out = _render(equi_t, xs, output_height, cx, cy, f_h, f_v, R_inv_t, sampling_method)
    if output_dtype == "uint8":
        out = torch.clamp(out, 0.0, 255.0).to(torch.uint8)
    if as_tensor:
        return out
    arr = out.detach().cpu().numpy()
    result = arr.astype(np.float32, copy=False) if output_dtype == "float32" else arr
    logger.info("Processing completed successfully")
    return result
