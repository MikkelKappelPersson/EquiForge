<p align="center">
    <img src="src/Logo+Name.svg" alt="EquiForge Logo" width="500"/>
</p>

<h4 align="center">
    A performant toolkit for equirectangular image processing and conversions
</h4>


<!--<img src=".img/equilib.png" alt="equilib" width="720"/>-->

<div align="center">
<a href="https://badge.fury.io/py/equiforge"><img src="https://badge.fury.io/py/equiforge.svg" alt="PyPI version"></a>
<a href="https://pypi.org/project/equiforge"><img src="https://img.shields.io/pypi/pyversions/equiforge"></a>
  <a href="https://github.com/MikkelKappelPersson/EquiForge/actions"><img src="https://github.com/MikkelKappelPersson/EquiForge/actions/workflows/python-package-tests.yml/badge.svg"></a>
  <a href="https://github.com/MikkelKappelPersson/EquiForge/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/mikkelkappelpersson/equiforge"></a>
</div>

## Features

- Convert perspective images to equirectangular projection (`pers2equi`)
- Convert equirectangular images to perspective view (`equi2pers`)
- GPU acceleration with CUDA (optional)

## Installation

### Prerequisites
- Python 3.8 or later
- numpy
- numba
- Pillow

### Using `pip`:

```bash
pip install equiforge
```

### CUDA GPU Support
To enable CUDA GPU support, install the [latest graphics drivers from NVIDIA](https://www.nvidia.com/en-us/drivers/) for your platform. Then install the CUDA Toolkit package.

For CUDA 12, cuda-nvcc and cuda-nvrtc are required:
```bash
$ conda install -c conda-forge cuda-nvcc cuda-nvrtc "cuda-version>=12.0"
```

## Example Usage

```python
from equiforge import pers2equi

# Convert perspective image to equirectangular
equi_image = pers2equi(
    'input.jpg',
    output_height=2048, 
    fov_x=90.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0
)
```

```python
from equiforge import pers2equi
# Convert equirectangular image to perspective view
pers_image = equi2pers(
    'equirectangular.jpg',
    output_width=1920,
    output_height=1080,
    fov_x=90.0,
    yaw=45.0,
    pitch=15.0,
    roll=0.0
)
```

## Documentation

For more examples and detailed documentation, see the Jupyter notebooks included in the repository.

## Acknowledgements:

- [equilib](https://github.com/haruishi43/equilib)
- [Perspective-and-Equirectangular](https://github.com/timy90022/Perspective-and-Equirectangular)
