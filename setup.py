# import setuptools  # important
from distutils.core import setup, Extension
import numpy as np

setup(name = 'myModule',
      version = '1.0',
      ext_modules = [Extension('myModule', ['test.c'],
                     include_dirs=[np.get_include()])])