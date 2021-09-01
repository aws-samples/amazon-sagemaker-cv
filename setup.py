#!/usr/bin/env python

import glob
import os
import copy
import torch
from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

requirements = ["torch", "torchvision"]


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "smcv_utils")

    main_file = glob.glob(os.path.join(extensions_dir, "vision.cpp"))
    main_file_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    source_cuda_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc", "*.cu"))
    source_cpp_nhwc = glob.glob(os.path.join(extensions_dir, "cuda/nhwc", "*.cpp"))

    sources = main_file + source_cpu
    sources_nhwc = source_cpp_nhwc + source_cuda_nhwc + main_file_nhwc
    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "smcv_utils._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        extension("smcv_utils.NHWC",
            sources_nhwc,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=copy.deepcopy(extra_compile_args),
        )
    ]

    return ext_modules


setup(
    name="smcv_utils",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/aws/smcv_utils",
    description="Computer vision PyTorch addons for Amazon SageMaker",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
