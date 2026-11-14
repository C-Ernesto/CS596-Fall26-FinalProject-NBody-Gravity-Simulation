"""Build the optional compiled hierarchical-interaction backend in place."""

from __future__ import annotations

import os

import numpy as np
from setuptools import Extension, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


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
        ),
        CUDAExtension(
            "ml._hierarchical_cuda",
            sources=[
                "ml/cpp/hierarchical_cuda.cpp",
                "ml/cpp/hierarchical_cuda_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--std=c++17"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
