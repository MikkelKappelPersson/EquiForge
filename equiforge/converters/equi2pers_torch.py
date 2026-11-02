"""PyTorch backend prototype for equirectangular -> perspective.

Vectorized alternative to the Numba CPU/CUDA paths. No custom kernels:
coordinate math is tensor ops, sampling is advanced-indexing bilinear with
explicit longitude wrap (x) + latitude clamp (y) — same semantics as
equi2pers_cpu_kernel (wrap_x=True).

Status: prototype for benchmarking on RTX 5090. Single view only, no batching yet.
"""
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

from ..utils.projection_utils import create_rotation_matrix, calculate_focal_length
from ..utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def _to_tensor(equi, device):
    """Convert (H,W,3) numpy or torch input to float32 torch tensor on device."""
    if HAS_TORCH is False:
        raise ImportError("torch is required for equi2pers_torch")
    if isinstance(equi, torch.Tensor):
        t = equi.to(device=device, dtype=torch.float32)
        if t.ndim != 3 or t.shape[2] != 3:
            raise ValueError(f"expected (H,W,3) tensor, got {tuple(t.shape)}")
        return t
    arr = np.asarray(equi)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) array, got {arr.shape}")
    return torch.as_tensor(arr, device=device, dtype=torch.float32)


def equi2pers_torch(equi, output_width, output_height,
                    fov_x=90.0, yaw=0.0, pitch=0.0, roll=0.0,
                    sampling_method="bilinear", output_dtype="float32",
                    device=None, as_tensor=False):
    """Torch-vectorized equirectangular to perspective.

    Parameters match equi2pers plus:
    - device: torch device string or torch.device. None = cuda if available else cpu.
    - as_tensor: if True return torch tensor on device, else numpy (default False
      to mirror the Numba API which returns numpy).

    output_dtype: "float32" (default here, raw HDR) or "uint8" (clip to [0,255]).
    """
    if not HAS_TORCH:
        raise ImportError("torch is not installed")
    if output_dtype not in ("uint8", "float32"):
        raise ValueError(f"output_dtype must be 'uint8' or 'float32', got {output_dtype!r}")
    if sampling_method not in ("bilinear", "nearest"):
        raise ValueError(f"sampling_method must be 'bilinear' or 'nearest', got {sampling_method!r}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    equi_t = _to_tensor(equi, dev)  # (H,W,3) float32
    equi_h, equi_w = equi_t.shape[0], equi_t.shape[1]

    fov_x = max(0.1, min(179.9, fov_x))
    cx = output_width // 2
    cy = output_height // 2
    fov_x_rad, yaw_rad, pitch_rad, roll_rad = map(np.radians, [fov_x, yaw, pitch, roll])
    f_h, f_v = calculate_focal_length(output_width, output_height, fov_x_rad)
    f_h = float(f_h)
    f_v = float(f_v)
    R = create_rotation_matrix(yaw_rad, pitch_rad, roll_rad)
    R_inv = np.linalg.inv(R).astype(np.float32)
    R_inv_t = torch.as_tensor(R_inv, device=dev, dtype=torch.float32)

    ys = torch.arange(output_height, device=dev, dtype=torch.float32)
    xs = torch.arange(output_width, device=dev, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H_out, W_out)

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
        y_clamped = torch.clamp(equi_y, 0.0, float(equi_h - 1))
        y_int = torch.round(y_clamped).long().clamp_(0, equi_h - 1)
        w_f = float(equi_w)
        x_wrapped = equi_x - torch.floor(equi_x / w_f) * w_f
        x_wrapped = torch.clamp(x_wrapped, 0.0, w_f - 1e-6) if equi_w > 1 else torch.zeros_like(x_wrapped)
        # guard fp edge x_wrapped == w
        x_wrapped = torch.where(x_wrapped >= w_f, torch.zeros_like(x_wrapped), x_wrapped)
        x_int = (torch.round(x_wrapped).long() % equi_w)
        out = equi_t[y_int, x_int]  # (H_out, W_out, 3)
    else:  # bilinear, wrap x / clamp y (matches CPU wrap_x=True)
        y_upper = float(equi_h) - 1.001
        if y_upper < 0.0:
            y_upper = 0.0
        y_c = torch.clamp(equi_y, 0.0, y_upper)
        y0 = torch.floor(y_c).long()
        y0 = y0.clamp(0, equi_h - 1)
        y1 = torch.clamp(y0 + 1, max=equi_h - 1)
        wy = (y_c - y0.float()).unsqueeze(-1)  # (H,W,1)

        w_f = float(equi_w)
        x_wrapped = equi_x - torch.floor(equi_x / w_f) * w_f
        x_wrapped = torch.where(x_wrapped >= w_f, torch.zeros_like(x_wrapped), x_wrapped)
        x_wrapped = torch.where(x_wrapped < 0.0, torch.zeros_like(x_wrapped), x_wrapped)
        x0 = torch.floor(x_wrapped).long().clamp(0, equi_w - 1)
        x1 = (x0 + 1) % equi_w
        wx = (x_wrapped - x0.float()).unsqueeze(-1)

        # advanced-index gathers -> (H_out, W_out, 3)
        top = equi_t[y0, x0] * (1.0 - wx) + equi_t[y0, x1] * wx
        bottom = equi_t[y1, x0] * (1.0 - wx) + equi_t[y1, x1] * wx
        out = top * (1.0 - wy) + bottom * wy

    if output_dtype == "uint8":
        out = torch.clamp(out, 0.0, 255.0).to(torch.uint8)

    if as_tensor:
        return out
    return out.detach().cpu().numpy()
