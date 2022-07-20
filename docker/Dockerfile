FROM nvcr.io/nvidia/pytorch:22.04-py3

ARG BUILD_ARTIFACTS=https://raw.githubusercontent.com/aws/deep-learning-containers/master/pytorch/training/docker/build_artifacts/
ARG SMD_DATA_PARALLEL_URL=https://smdataparallel.s3.amazonaws.com/binary/pytorch/1.12.0/cu116/2022-07-19/smdistributed_dataparallel-1.5.0-cp38-cp38-linux_x86_64.whl
ARG EFA_INSTALLER_VERSION=latest
ARG AWS_OFI_NCCL_VERSION=aws
ARG NCCL_TESTS_VERSION=master
ARG NCCL_VERSION=v2.11.4-1
ARG OPEN_MPI_PATH=/opt/amazon/openmpi
ARG EFA_PATH=/opt/amazon/efa

ENV NCCL_VERSION=v2.11.4-1
ENV EFA_VERSION=1.15.1
ENV OMPI_VERSION=4.1.1
ENV BRANCH_OFI=1.2.0-aws
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
  && git clone https://github.com/NVIDIA/nccl.git -b ${NCCL_VERSION} \
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

###################################################
## Install NCCL-tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && git checkout ${NCCL_TESTS_VERSION} \
    && make MPI=1 \
       MPI_HOME=/opt/amazon/openmpi/ \
       CUDA_HOME=/usr/local/cuda \
       NCCL_HOME=/opt/nccl/build \
       NVCC_GENCODE="-gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60"

RUN rm -rf /var/lib/apt/lists/*
ENV LD_PRELOAD=/usr/local/lib/libnccl.so

RUN pip install --no-cache-dir -U \
    # https://github.com/asweigart/pyautogui/issues/706
    opencv-python==4.5.5.64 \ 
    numpy \
    seaborn \
    ipython \
    pyyaml \
    awscli \
    boto3 \
    packaging \
    psutil \
    Pillow \
    scipy \
    pybind11 \
    click \
    mpi4py \
    cmake \
    torchnet \
    cryptography \
    wget \
    ujson \
    yacs
    
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
  
# Install Nvidia's accelerated PyCOCOTools
RUN git clone https://github.com/NVIDIA/cocoapi && \
    cd cocoapi/PythonAPI && \
    pip install -e . --verbose
    
# Needed for CV2
RUN apt update && apt install -y ffmpeg libsm6 libxext6 && \
    apt-get install -y libxrender-dev
    
# Add studio utils
RUN pip install ipykernel && \
    python -m ipykernel install --sys-prefix && \
    pip install jupyter_kernel_gateway
    
# Install Mish cuda - need to move CUDAApplyUtils.cuh https://github.com/thomasbrandon/mish-cuda/issues/9
RUN git clone https://github.com/thomasbrandon/mish-cuda && \
    cd mish-cuda/ && \
    cp external/CUDAApplyUtils.cuh csrc/ && \
    pip install .
    
# Add PT S3 Plugin
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

# Install libboost from source. This package is needed for smdataparallel functionality [for networking asynchronous IO].
RUN wget https://sourceforge.net/projects/boost/files/boost/1.73.0/boost_1_73_0.tar.gz/download -O boost_1_73_0.tar.gz \
 && tar -xzf boost_1_73_0.tar.gz \
 && cd boost_1_73_0 \
 && ./bootstrap.sh \
 && ./b2 threading=multi --prefix=/opt/conda -j 64 cxxflags=-fPIC cflags=-fPIC install || true \
 && cd .. \
 && rm -rf boost_1_73_0.tar.gz \
 && rm -rf boost_1_73_0 \
 && cd /opt/conda/include/boost

RUN chmod +x /usr/local/bin/start_with_right_hostname.sh

# This build of SMDDP isn't working in the NGC container, need to build another
# Install SM Distributed DataParallel binary
#RUN SMDATAPARALLEL_PT=1 pip install --no-cache-dir ${SMD_DATA_PARALLEL_URL}
#ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.8/site-packages/smdistributed/dataparallel/lib/:$LD_LIBRARY_PATH"

# Clone YoloV5 repo
RUN git clone https://github.com/HerringForks/SMDDP-Examples/ \
    && mv SMDDP-Examples/pytorch/object_detection/yolov5 . \
    && rm -r SMDDP-Examples

ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]
