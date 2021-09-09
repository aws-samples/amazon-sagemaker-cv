#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

requirements = ["tensorflow"]

setup(
    name="sagemakercv",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/aws-samples/amazon-sagemaker-cv",
    description="Computer vision in TensorFlow with Amazon Sagemaker",
    packages=find_packages(exclude=("configs", "tests")),
)