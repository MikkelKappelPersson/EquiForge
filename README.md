# EquiForge

A toolkit for equirectangular image processing and conversion.

## Features

- Convert perspective images to equirectangular projection (`pers2equi`)
- Convert equirectangular images to perspective view (`equi2pers`)
- GPU acceleration with CUDA (optional)

## Installation

Basic installation:
```bash
pip install equiforge
```

With visualization support:
```bash
pip install equiforge[viz]
```

With CUDA support:
```bash
pip install equiforge[cuda]
```

With development tools:
```bash
pip install equiforge[dev]
```

## Requirements

- Python 3.7+
- For CUDA support: CUDA Toolkit 12.x

## Example Usage

```python
from equiforge import pers2equi, equi2pers

# Convert perspective image to equirectangular
equi_image = pers2equi(
    'input.jpg',
    output_height=2048,
    fov_h=90.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0
)

# Convert equirectangular image to perspective view
pers_image = equi2pers(
    'equirectangular.jpg',
    output_width=1920,
    output_height=1080,
    fov_h=90.0,
    yaw=45.0,
    pitch=15.0,
    roll=0.0
)
```

## Documentation

For more examples and detailed documentation, see the Jupyter notebooks included in the repository.
