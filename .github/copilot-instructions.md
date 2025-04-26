# Project Description

EquiForge is a performant Python toolkit for equirectangular image processing and projection conversions. The library provides tools for:

1. Converting between various image projections, including perspective and equirectangular formats
2. Processing and manipulating 360° imagery
3. Supporting both CPU and GPU acceleration via CUDA and numba

The project prioritizes performance and ease of use for handling 360° imagery and different projection types.

# Code Practices

- Keep code simple and easily readable
- Avoid unnecessary complexity
- Avoid exception handling where possible; prefer to let code fail fast and explicitly
- use float32 for all image processing operations

