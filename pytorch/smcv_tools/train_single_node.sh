#!/usr/bin/env bash

GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
CONFIG=$1
REPO=$2
REGION=${3-us-west-2}
AWS_ACCOUNT=${4-`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`}
IMAGE_NAME=${5-smcv-pt-1.8}

docker run --rm -it --gpus all --name smcv --net=host --uts=host --ipc=host \
    --ulimit stack=67108864 \
    --ulimit memlock=-1 \
    --ulimit nofile=131072 \
    --security-opt=seccomp=unconfined \
    -v ~/SageMaker/data/:/workspace/data/ \
    -v ~/SageMaker/sagemakercv/:/workspace/sagemakercv/ \
    ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME} \
    /bin/bash -c "ls && \
    cd /workspace/sagemakercv/pytorch/tools/ && \
    python -m torch.distributed.launch \
    --nnodes 1 \
    --node_rank 0 \
    --nproc_per_node ${GPU_COUNT} \
    train.py \
    --config ${CONFIG}"






