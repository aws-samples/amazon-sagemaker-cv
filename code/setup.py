#!/usr/bin/env python
from setuptools import setup, find_packages

install_requires = ["pytorch-lightning",
                    "lightning-bolts",]

setup(
    name="sm_resnet",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/johnbensnyder/sagemaker_lightning",
    description="Resnet test",
    packages=find_packages(),
    install_requires=install_requires,
)