#!/usr/bin/env python

import glob
import os
import copy
import torch
from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision"]

setup(
    name="sagemakercv",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/aws-samples/amazon-sagemaker-cv",
    description="Computer vision in Pytorch with Amazon Sagemaker",
    packages=find_packages(exclude=("configs", "tests", "tools")),
)
