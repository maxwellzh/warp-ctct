import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if not torch.cuda.is_available():
    raise Exception("CPU version is not implemented")


requirements = [
    "pybind11",
    "torch>=1.11.0"
]

setup(
    name="warp_ctct",
    version="0.3.0",
    description="PyTorch bindings for CUDA-Warp CTC-Transducer",
    url="https://github.com/maxwellzh/warp-ctct",
    author="Huahuan Zheng",
    author_email="maxwellzh@outlook.com",
    license="MIT",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="warp_ctct._C",
            sources=[
                "csrc/binding.cpp",
                "csrc/gather.cu",
                "csrc/logmatmul.cu",
                "csrc/simple.cu",
                "csrc/simple_unnorm.cu"
            ]
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    setup_requires=requirements,
    install_requires=requirements
)
