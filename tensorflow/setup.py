#!/usr/bin/env python

import sys
py_version = f"{sys.version_info.major}{sys.version_info.minor}"

from setuptools import find_packages
from setuptools import setup

coco_tools = "git+https://github.com/NVIDIA/cocoapi.git#subdirectory=PythonAPI"

install_requires = ["tensorflow_addons",
                    "tensorflow_datasets",
                    "yacs",
                    "matplotlib",
                    "mpi4py",
                    "opencv-python",
                    "pycocotools",
                    "tensorflow-io"]



setup(
    name="sagemakercv",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/aws-samples/amazon-sagemaker-cv",
    description="Computer vision in TensorFlow with Amazon Sagemaker",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=install_requires
)
