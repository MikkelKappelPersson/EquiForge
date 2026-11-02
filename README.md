<p align="center">
    <img src="src/Logo+Name.svg" alt="EquiForge Logo" width="500"/>
</p>

<h4 align="center">
    A precision toolkit for equirectangular image processing and conversions
</h4>

<div align="center">
<a href="https://badge.fury.io/py/equiforge"><img src="https://badge.fury.io/py/equiforge.svg" alt="PyPI version"></a>
<a href="https://pypi.org/project/equiforge"><img src="https://img.shields.io/pypi/pyversions/equiforge"></a>
  <a href="https://github.com/MikkelKappelPersson/EquiForge/actions"><img src="https://github.com/MikkelKappelPersson/EquiForge/actions/workflows/python-package-tests.yml/badge.svg"></a>
  <a href="https://github.com/MikkelKappelPersson/EquiForge/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/mikkelkappelpersson/equiforge"></a>
</div>

## Features

- Convert perspective images to equirectangular projection (`pers2equi`)
- Convert equirectangular images to perspective views (`equi2pers`)
- Torch-only backend: same vectorized code runs on CPU and CUDA — no Numba, no custom kernels, no toolkit setup
- HDR-safe: `output_dtype="float32"` returns raw radiance (no `[0,255]` clip) for EXR training corpora; `"uint8"` (default) preserves legacy clipped output
- True bilinear (sub-pixel), periodic longitude seam (last↔first blend), clamped poles

## Installation

Prerequisites: Python 3.10+, `numpy`, `torch>=2.4`, `Pillow`.

```bash
pip install equiforge
# GPU wheels (e.g. CUDA 12.8 for RTX 5090) come with torch itself:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

No `cuda-nvcc` / `CUDA_HOME` / `libdevice` setup needed.

## Example Usage

```python
from equiforge import pers2equi, equi2pers

# Perspective -> equirectangular (2:1 output)
equi = pers2equi('input.jpg', output_height=1024, fov_x=90.0)

# Equirectangular -> perspective (HDR, GPU)
pers = equi2pers(
    'pano.exr',  # float32 radiance, values >> 255 supported
    output_width=512, output_height=512,
    fov_x=90.0, yaw=45.0, pitch=0.0, roll=0.0,
    use_gpu=True, output_dtype="float32",
)
```

Training-corpus pattern — upload once, render many views on GPU:

```python
import torch
from equiforge import equi2pers

pano = torch.as_tensor(exr_array, dtype=torch.float32, device="cuda")  # (H,W,3)
for yaw in [0, 90, 180, 270]:
    view = equi2pers(pano, 512, 512, fov_x=90.0, yaw=yaw,
                     output_dtype="float32", device="cuda", as_tensor=True)
    # view is a (512,512,3) cuda tensor — feed straight to your loader
```

API notes:
- `output_dtype="uint8"` (default): clip to `[0,255]`, legacy behaviour.
- `output_dtype="float32"`: raw samples, no clip/cast.
- `sampling_method="bilinear"` (default) or `"nearest"`.
- `device=None` auto-selects (`cuda` if `use_gpu and torch.cuda.is_available()` else `cpu`); pass `device="cpu"/"cuda"` to override.
- `as_tensor=True` returns a torch tensor on `device` instead of numpy.

## Documentation

For more examples and detailed documentation, see the Jupyter notebooks included in the repository.

## Acknowledgements:

- [equilib](https://github.com/haruishi43/equilib)
- [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular)
