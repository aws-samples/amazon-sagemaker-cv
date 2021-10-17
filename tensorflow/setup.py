#!/usr/bin/env python

import sys
py_version = f"{sys.version_info.major}{sys.version_info.minor}"

from setuptools import find_packages
from setuptools import setup

if py_version=="37":
    pycocotools_whl = "https://aws-smcv-us-west-2.s3.us-west-2.amazonaws.com/utils/binaries/pycocotools-2.0%2Bnv0.6.0-cp37-cp37m-linux_x86_64.whl"
else:
    pycocotools_whl = "https://aws-smcv-us-west-2.s3.us-west-2.amazonaws.com/utils/binaries/pycocotools-2.0%2Bnv0.6.0-cp38-cp38-linux_x86_64.whl"
    

install_requires = ["tensorflow_addons",
                    "tensorflow_datasets",
                    "yacs",
                    "matplotlib",
                    "mpi4py",
                    "opencv-python",
                    f"pycocotools @ {pycocotools_whl}"]



setup(
    name="sagemakercv",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/aws-samples/amazon-sagemaker-cv",
    description="Computer vision in TensorFlow with Amazon Sagemaker",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=install_requires
)