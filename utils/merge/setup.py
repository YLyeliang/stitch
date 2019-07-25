from setuptools import setup
from Cython.Build import cythonize

setup(name='merge',ext_modules=cythonize('src/merge.pyx'))