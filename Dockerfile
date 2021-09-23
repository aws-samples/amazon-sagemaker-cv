ARG REGION=us-west-2
ARG DLC_ACCOUNT=763104351884
ARG FRAMEWORK=pytorch-training
ARG VERSION=1.9.0-gpu-py38-cu111-ubuntu20.04
FROM ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${FRAMEWORK}:${VERSION}

RUN pip install \
    yacs \
    matplotlib \
    imgaug 

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
