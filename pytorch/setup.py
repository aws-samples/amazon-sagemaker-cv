#!/usr/bin/env python
import sys
py_version = f"{sys.version_info.major}{sys.version_info.minor}"

from setuptools import find_packages
from setuptools import setup

import torch

torch_version = ''.join(torch.__version__.split('.')[:2])

if py_version=="36":
    pycocotools_whl = "https://sagemakercv.s3.us-west-2.amazonaws.com/cocoapi/pycocotools-2.0%2Bnv0.6.0-cp36-cp36m-linux_x86_64.whl"
    awsio_whl = "https://aws-s3-plugin.s3-us-west-2.amazonaws.com/binaries/0.0.1/93fdaed/awsio-0.0.1-cp36-cp36m-manylinux1_x86_64.whl"
else:
    pycocotools_whl = "https://sagemakercv.s3.us-west-2.amazonaws.com/cocoapi/pycocotools-2.0%2Bnv0.6.0-cp38-cp38-linux_x86_64.whl"
    awsio_whl = "https://aws-s3-plugin.s3.us-west-2.amazonaws.com/binaries/0.0.1/1c3e69e/awsio-0.0.1-cp38-cp38-manylinux1_x86_64.whl"
    
if torch_version=="16":
    smcv_utils_whl = "https://sagemakercv.s3.us-west-2.amazonaws.com/utils/pt-1.6/smcv_utils-0.0.1-cp36-cp36m-linux_x86_64.whl"
elif torch_version=="17":
    smcv_utils_whl = "https://sagemakercv.s3.us-west-2.amazonaws.com/utils/pt-1.7/smcv_utils-0.0.1-cp36-cp36m-linux_x86_64.whl"
elif torch_version=="18":
    smcv_utils_whl = "https://sagemakercv.s3.us-west-2.amazonaws.com/utils/pt-1.8/smcv_utils-0.0.1-cp36-cp36m-linux_x86_64.whl"
else:
    smcv_utils_whl = "https://sagemakercv.s3.us-west-2.amazonaws.com/utils/pt-1.9/smcv_utils-0.0.1-cp38-cp38-linux_x86_64.whl"

install_requires = ["yacs", 
                    "matplotlib",
                    "mpi4py",
                    "opencv-python",
                    f"pycocotools @ {pycocotools_whl}",
                    f"smcv_utils @ {smcv_utils_whl}",
                    f"awsio @ {awsio_whl}"]

setup(
    name="sagemakercv",
    version="0.1",
    author="jbsnyder",
    url="https://github.com/aws-samples/amazon-sagemaker-cv",
    description="Computer vision in Pytorch with Amazon Sagemaker",
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=install_requires,
)
