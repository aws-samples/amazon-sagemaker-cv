ARG REGION=us-west-2
ARG DLC_ACCOUNT=763104351884
ARG FRAMEWORK=pytorch-training
ARG VERSION=1.9.0-gpu-py38-cu111-ubuntu20.04
FROM ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${FRAMEWORK}:${VERSION}

# set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update

# install CV2 and python tools
RUN apt-get update

RUN apt-get install -y python3-opencv numactl

RUN pip install \
    ninja \
    yacs \
    cython \
    matplotlib \
    imgaug \
    tqdm \
    numba \
    opencv-python==3.4.11.45 \
    pybind11 \
    gpustat \
    mpi4py
RUN pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip

# install pycocotools
RUN git clone https://github.com/NVIDIA/cocoapi && \
    cd cocoapi/PythonAPI && \
    pip install -v --no-cache-dir -e .
    
# add kernel tools to use interactively with studio
RUN pip install ipykernel jupyterlab && \
    python -m ipykernel install --sys-prefix && \
    pip install jupyter_kernel_gateway

# Starts framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]
