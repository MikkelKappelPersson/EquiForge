from setuptools import setup, find_packages

setup(
    name="equiforge",
    use_scm_version=True,
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "numba>=0.53.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        'viz': ["matplotlib>=3.4.0"],  # For visualization and notebooks
        'dev': ["pytest>=6.0.0", "jupyter>=1.0.0", "matplotlib>=3.4.0"],  # Development tools
        'cuda': ["cupy-cuda12x>=12.0.0"],  # GPU acceleration with CUDA
        # Note: CUDA toolkit (nvcc, nvrtc) must be installed separately
    },
    setup_requires=[
        "setuptools_scm>=6.0.1",
    ],
    author="Mikkel Kappel Persson",
    author_email="mikkelkp@hotmail.com",
    description="A toolkit for equirectangular image processing",
    keywords="equirectangular, panorama, image processing, 360",
    url="https://github.com/mikkelkappelpersson/equiforge",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)
