# EquiForge

EquiForge is a tool for transforming perspective images into equirectangular format, useful for creating panoramic and 360° content.

## Features

- Convert standard perspective images to equirectangular format
- Adjust field-of-view (FOV), yaw, pitch, and roll parameters
- Optimized performance using multi-threading and GPU acceleration
- Simple command-line interface for batch processing

## Installation

```bash
# Clone this repository
git clone https://github.com/mikkelkappelpersson/EquiForge.git
cd EquiForge

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python persp2equir.py input_image.jpg --output output_image.jpg --fov 90 --yaw 60 --pitch 0 --roll 0
```

### Python Import

```python
from persp2equir import perspective_to_equirectangular

result = perspective_to_equirectangular(
    'input_image.jpg',
    output_height=4096,
    fov_h=90.0,
    yaw=60.0,
    pitch=0.0,
    roll=0.0
)
```

## Parameter Reference

- **FOV**: Standard DSLR (60-90°), Wide-angle (100-120°), Smartphone (70-80°)
- **Yaw**: Left/right rotation (0° = forward)
- **Pitch**: Up/down rotation (0° = level, positive = up)
- **Roll**: Clockwise/counterclockwise rotation

## Performance Optimization

To improve processing speed, EquiForge is optimized to use:

1. **Multi-threading**: Uses multiple CPU cores in parallel
2. **Numba JIT Compilation**: Just-in-time compilation for CPU code
3. **Numba CUDA**: GPU acceleration that works with most NVIDIA GPUs

## Benefits

1. **Portability**: Process images without opening the Jupyter notebook
2. **Batch Processing**: Can be used in shell scripts to process multiple images
3. **Integration**: Can be imported into other Python projects
4. **Simplicity**: Clean interface with sensible defaults

## License

[MIT License](LICENSE)
