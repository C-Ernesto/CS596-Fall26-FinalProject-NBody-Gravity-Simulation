"""Build the optional compiled hierarchical-interaction backend in place."""

from __future__ import annotations

import os

import numpy as np
from setuptools import Extension, setup


if os.name == "nt":
    compile_arguments = ["/O2", "/std:c++17", "/openmp"]
    link_arguments: list[str] = []
else:
    compile_arguments = ["-O3", "-std=c++17", "-fopenmp"]
    link_arguments = ["-fopenmp"]


setup(
    name="nbody-hierarchical-backend",
    version="0.1.0",
    ext_modules=[
        Extension(
            "ml._hierarchical_cpp",
            sources=["ml/cpp/hierarchical_interactions.cpp"],
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=compile_arguments,
            extra_link_args=link_arguments,
        )
    ],
)
