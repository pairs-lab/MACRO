from setuptools import Extension, setup
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)

setup(
    name='marco-grid-geometry', version='0.1.0',
    ext_modules=[Extension('_grid_geometry_native', ['geometry.cpp'],
                           language='c++', extra_compile_args=['-O3', '-std=c++17',
                                                               '-fno-fast-math', '-ffp-contract=off'])],
)
