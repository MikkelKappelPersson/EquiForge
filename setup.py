from setuptools import setup, find_packages

setup(
    name="equiforge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "Pillow>=8.0.0",
        "numba>=0.53.0",
    ],
    entry_points={
        'console_scripts': [
            'pers2equi=equiforge.cli:main',
        ],
    },
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
