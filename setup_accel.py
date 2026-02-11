"""
Build gaddag_accel Cython extension.

Usage:
    python setup_accel.py build_ext --inplace

Requires:
    pip install cython
    
On Windows: Visual Studio Build Tools
On Linux: gcc
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import sys

extra_compile_args = []
if sys.platform != 'win32':
    extra_compile_args = ['-O3', '-march=native']
else:
    extra_compile_args = ['/O2']

setup(
    name='gaddag_accel',
    ext_modules=cythonize([
        Extension(
            'gaddag_accel',
            sources=['gaddag_accel.pyx'],
            extra_compile_args=extra_compile_args,
        ),
    ]),
)
