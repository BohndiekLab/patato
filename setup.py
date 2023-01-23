#  Copyright (c) Thomas Else 2023.
#  License: MIT

import pybind11
import setuptools
from setuptools import Extension

modelbased_c = Extension("patato.recon.model_based.generate_model",
                         sources=["patato/recon/model_based/generate_model.cpp"],
                         language="c++", extra_compile_args=["-std=c++11"],
                         include_dirs=[pybind11.get_include()], )
modelbased_c2 = Extension("patato.recon.model_based.generate_model_refraction",
                          sources=["patato/recon/model_based/generate_model_refraction.cpp"],
                          language="c++", extra_compile_args=["-std=c++11"],
                          include_dirs=[pybind11.get_include()], )

setuptools.setup(
    ext_modules=[modelbased_c2, modelbased_c]
)
