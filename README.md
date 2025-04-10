# EquiForge

A toolkit for equirectangular image processing and format conversion.

## Features

- Convert perspective images to equirectangular projection
- Optimized processing using GPU (CUDA) when available
- Multi-core CPU processing as fallback

## Installation

```bash
# Install from source
pip install -e .
```

## Usage

### Command Line

```bash
# Basic usage
persp2equir input.jpg output.jpg

# With options
persp2equir input.jpg output.jpg --fov 120 --height 2048 --yaw 30 --pitch 15 --roll 0 --cpu
```

### Python API

```python
from equiforge.converters.persp2equir import perspective_to_equirectangular
from PIL import Image
import numpy as np

# Load an image
img = np.array(Image.open('input.jpg'))

# Convert to equirectangular
equirect = perspective_to_equirectangular(
    img,
    output_height=2048,
    fov_h=90.0,
    yaw=0.0,
    pitch=0.0,
    roll=0.0,
    use_gpu=True
)

# Save the result
Image.fromarray(equirect).save('output.jpg')
```

## License

MIT
