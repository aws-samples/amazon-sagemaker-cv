FROM nvcr.io/nvidia/pytorch:22.04-py3

ARG NCCL_VERSION=v2.12.10-1
ARG OPEN_MPI_PATH=/opt/amazon/openmpi
ARG EFA_PATH=/opt/amazon/efa
ENV EFA_VERSION=1.16.0
ENV OMPI_VERSION=4.1.2
ENV BRANCH_OFI=1.3.0-aws
# TODO: Add DGL environment variables back when DGL is re-enabled
# ENV DGLBACKEND=pytorch
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV MANUAL_BUILD=0
ENV RDMAV_FORK_SAFE=1

# Clean existing Nvidia libraries
RUN apt-get update -y
RUN apt-get remove -y --allow-change-held-packages \
                      libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1 \
                      libnccl2 libnccl-dev
RUN rm -rf /opt/hpcx \
    && rm -rf /usr/local/mpi \
    && rm -rf /usr/local/ucx \
    && rm -f /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig
ENV OPAL_PREFIX=

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    git \
    gcc \
    vim \
    kmod \
    openssh-client \
    openssh-server \
    build-essential \
    curl \
    autoconf \
    libtool \
    gdb \
    automake \
    cmake \
    apt-utils \
    ninja-build

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python /tmp/get-pip.py \
    && pip install awscli pynvml

###################################################
## Install NCCL
RUN cd /tmp \
  && git clone https://github.com/NVIDIA/nccl.git -b v2.12.10-1 \
  && cd nccl \
  && make -j64 src.build BUILDDIR=/usr/local \
  && rm -rf /tmp/nccl
    
# Install EFA along without AWS OPEN_MPI
RUN mkdir /tmp/efa \
  && cd /tmp/efa \
  && curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz \
  && tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz \
  && cd aws-efa-installer \
  && apt-get update \
  && ./efa_installer.sh -y --skip-kmod -g \
  && rm -rf $OPEN_MPI_PATH \
  && rm -rf /tmp/efa \
  && rm -rf /tmp/aws-efa-installer-${EFA_VERSION}.tar.gz \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

# Install OpenMPI without libfabric support
RUN mkdir /tmp/openmpi && \
  cd /tmp/openmpi && \
  wget --quiet https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.gz && \
  tar zxf openmpi-${OMPI_VERSION}.tar.gz && \
  cd openmpi-${OMPI_VERSION} && \
  ./configure --enable-orterun-prefix-by-default --prefix=$OPEN_MPI_PATH --with-cuda && \
  make -j $(nproc) all && \
  make install && \
  ldconfig && \
  cd / && \
  rm -rf /tmp/openmpi

ENV PATH="$OPEN_MPI_PATH/bin:$PATH"
ENV LD_LIBRARY_PATH=$OPEN_MPI_PATH/lib/:$EFA_PATH/lib/:$LD_LIBRARY_PATH

# Configure Open MPI and configure NCCL parameters
RUN mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real \
 && echo '#!/bin/bash' > $OPEN_MPI_PATH/bin/mpirun \
 && echo "${OPEN_MPI_PATH}/bin/mpirun.real --allow-run-as-root \"\$@\"" >> $OPEN_MPI_PATH/bin/mpirun \
 && chmod a+x $OPEN_MPI_PATH/bin/mpirun \
 && echo "hwloc_base_binding_policy = none" >> $OPEN_MPI_PATH/etc/openmpi-mca-params.conf \
 && echo "rmaps_base_mapping_policy = slot" >> $OPEN_MPI_PATH/etc/openmpi-mca-params.conf \
 && echo NCCL_DEBUG=INFO >> /etc/nccl.conf \
 && echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Install AWS OFI NCCL plug-in
RUN apt-get update && apt-get install -y autoconf
RUN mkdir /tmp/efa-ofi-nccl \
  && cd /tmp/efa-ofi-nccl \
  && git clone https://github.com/aws/aws-ofi-nccl.git -b v${BRANCH_OFI} \
  && cd aws-ofi-nccl \
  && ./autogen.sh \
  && ./configure --with-libfabric=/opt/amazon/efa \
  --with-mpi=/opt/amazon/openmpi \
  --with-cuda=/usr/local/cuda \
  --with-nccl=/usr/local --prefix=/usr/local \
  && make \
  && make install \
  && rm -rf /tmp/efa-ofi-nccl \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

# Install OpenSSH for MPI to communicate between containers, allow OpenSSH to talk to containers without asking for confirmation
RUN apt-get update \
 && apt-get install -y  --allow-downgrades --allow-change-held-packages --no-install-recommends \
 && apt-get install -y --no-install-recommends openssh-client openssh-server \
 && mkdir -p /var/run/sshd \
 && cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
 && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
 && mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Configure OpenSSH so that nodes can communicate with each other
RUN mkdir -p /var/run/sshd && \
 sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
 mkdir -p /root/.ssh/ && \
 ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
 cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
 && printf "Host *\n StrictHostKeyChecking no\n" >> /root/.ssh/config

RUN rm -rf /var/lib/apt/lists/*
ENV LD_PRELOAD=/usr/local/lib/libnccl.so

ARG BUILD_ARTIFACTS=https://raw.githubusercontent.com/aws/deep-learning-containers/master/pytorch/training/docker/build_artifacts/

RUN pip install --no-cache-dir -U \
    opencv-python \
    numpy \
    ipython \
    pyyaml \
    "awscli<2" \
    boto3 \
    packaging \
    psutil \
    "Pillow>=9.0.0" \
    scipy \
    pybind11 \
    click \
    mpi4py==3.0.3 \
    cmake==3.18.2.post1 \
    torchnet \
    "cryptography>3.2" \
    wget
    
# Copy workaround script for incorrect hostname
RUN wget ${BUILD_ARTIFACTS}/changehostname.c -O /changehostname.c
RUN wget ${BUILD_ARTIFACTS}/start_with_right_hostname.sh -O /usr/local/bin/start_with_right_hostname.sh
    
RUN pip install --no-cache-dir -U \
    smclarify \
    "sagemaker>=2,<3" \
    sagemaker-experiments==0.* \
    sagemaker-pytorch-training
    
# Install smdebug from souce
RUN cd /tmp \
  && git clone https://github.com/awslabs/sagemaker-debugger \
  && cd sagemaker-debugger \
  && python setup.py install \
  && rm -rf /tmp/* 

# Install Nvidia MRCNN codebase
WORKDIR /tmp

RUN git clone https://github.com/mlcommons/training_results_v2.0 && \
    mkdir -p /workspace/maskrcnn/ && \
    mv /tmp/training_results_v2.0/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/maskrcnn /workspace/ && \
    rm -rf /tmp/training_results_v2.0

WORKDIR /workspace

RUN cd maskrcnn/ && \
    pip install -e . --verbose

RUN git clone https://github.com/NVIDIA/cocoapi && \
    cd cocoapi/PythonAPI && \
    pip install -e . --verbose

# Install pip packages
RUN pip install git+https://github.com/mlcommons/logging.git@2.0.0-rc4 && \
    pip install ujson opencv-python==3.4.8.29 yacs
    
RUN apt update && apt install -y libsm6 libxext6 && \
    apt-get install -y libxrender-dev

# Add studio utils
RUN pip install ipykernel && \
    python -m ipykernel install --sys-prefix && \
    pip install jupyter_kernel_gateway

# Add PyTorch S3 Plugin
RUN apt-get update && apt-get install -y curl libcurl4-openssl-dev libssl-dev ca-certificates
RUN git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp && \
    cd aws-sdk-cpp/ && \
    mkdir sdk-build && \
    cd sdk-build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_ONLY="s3;transfer" && \
    make && \
    make install && \
    pip install pybind11 && \
    export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/opt/conda/lib/python3.8/site-packages/pybind11 && \
    cd ../.. && \
    git clone --recurse-submodules https://github.com/aws/amazon-s3-plugin-for-pytorch && \
    cd amazon-s3-plugin-for-pytorch && \
    pip install -v .
    
RUN chmod +x /usr/local/bin/start_with_right_hostname.sh

ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]
