ARG DLC_ACCOUNT=763104351884
ARG REGION=us-east-1
ARG DLC_REPO=tensorflow-training
ARG DLC_TAG=2.4.1-gpu-py37-cu110-ubuntu18.04

FROM ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${DLC_REPO}:${DLC_TAG}

# set timezone so opencv doesn't ask
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update

RUN apt update && apt install -y python3-opencv libopenblas-base \
        libomp-dev build-essential
        
RUN pip install --upgrade tensorflow_addons \
                          tensorflow_datasets \
                          yacs \
                          gpustat \
                          pybind11 \
                          mpi4py \
                          tqdm \
                          ipywidgets

RUN git clone https://github.com/johnbensnyder/cocoapi && \
        cd cocoapi/PythonAPI && \
        pip install -v --no-cache-dir -e .

# add kernel tools to use interactively with studio
RUN pip install ipykernel && \
    python -m ipykernel install --sys-prefix && \
    pip install jupyter_kernel_gateway
        
CMD ["bin/bash"]