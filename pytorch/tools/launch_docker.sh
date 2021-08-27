GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

REPO=$1
REGION=${2-us-west-2}
AWS_ACCOUNT=${3-`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`}
IMAGE_NAME=${4-smcv-pt-1.8}

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker run --rm -it -d --gpus all --name smcv --net=host --uts=host --ipc=host \
    --ulimit stack=67108864 \
    --ulimit memlock=-1 \
    --ulimit nofile=131072 \
    --security-opt=seccomp=unconfined \
    -v ~/data/:/workspace/data/ \
    -v ~/sagemakercv/:/workspace/sagemakercv/ \
    ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME} \
    /bin/bash
