FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04-sagemaker
  
ENV CUDNN_VERSION=8.2.0.53

RUN apt-key del 7fa2af80

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb

RUN dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update \
 # TODO: Remove systemd upgrade once it is updated in base image
 && apt-get -y upgrade --only-upgrade systemd \
 && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.3 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.3

COPY smcv_fork /amazon-sagemaker-cv

RUN pip install -v /amazon-sagemaker-cv/pytorch/cuda_utils
RUN git clone https://github.com/NVIDIA/cocoapi \
  && cd cocoapi/PythonAPI \
  && pip install -e .
