FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.5.0-gpu-py37-cu112-ubuntu18.04

RUN pip uninstall -y \
    smdistributed-dataparallel \
    smdistributed-modelparallel \
    horovod \
    tensorflow-estimator \
    tensorflow-gpu

RUN pip install https://aws-tensorflow-binaries.s3-us-west-2.amazonaws.com/tensorflow/r2.5_aws/20210820_112241/gpu/cu112/py37/tensorflow_gpu-2.5.1-cp37-cp37m-manylinux2010_x86_64.whl

RUN HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install horovod==0.23.0

COPY herring /herring

RUN cd /herring \
    && export CUDACXX=/usr/local/cuda/bin/nvcc \
    && export LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
    && SMDATAPARALLEL_EC2_ENABLED=1 SMDATAPARALLEL_TF=1 python setup.py bdist_wheel \
    && pip install dist/smdistributed_dataparallel-*.whl \
    && rm -rf /herring

RUN pip install --upgrade tensorflow_addons \
                          tensorflow_datasets \
                          yacs \
                          'git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI'