#!/usr/bin/env bash

GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
CONFIG=$1
REPO=$2
REGION=${3-us-west-2}
AWS_ACCOUNT=${4-`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`}
IMAGE_NAME=${5-smcv-tf-2.4}

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker run --rm -it --gpus all --name smcv --net=host --uts=host --ipc=host \
	--privileged --ulimit=stack=67108864 --ulimit=memlock=-1 --security-opt=seccomp=unconfined \
	-v ~/data/:/workspace/data/ \
	-v ~/sagemakercv/:/workspace/sagemakercv/ \
	${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME} \
    /bin/bash -c "cd /workspace/sagemakercv/tensorflow/tools/ && \
    mpirun --allow-run-as-root --tag-output --mca plm_rsh_no_tree_spawn 1 \
    --mca btl_tcp_if_exclude lo,docker0 \
    -np $GPU_COUNT -H localhost:$GPU_COUNT \
    -x NCCL_DEBUG=VERSION \
    -x LD_LIBRARY_PATH \
    -x PATH \
    --oversubscribe \
    python train.py --config ${CONFIG}"
