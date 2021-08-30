REPO=$1
REGION=${2-us-west-2}
AWS_ACCOUNT=${3-`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`}
IMAGE_NAME=${4-smcv-pt-1.8}
DLC_ACCOUNT=763104351884
CUDNN_SOURCE_IMAGE=nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p $BASEDIR/cudnn_headers/include
mkdir -p $BASEDIR/cudnn_headers/include_v8

docker run -it --rm -v $BASEDIR/cudnn_headers:/workspace/cudnn_headers ${CUDNN_SOURCE_IMAGE} /bin/bash -c "cp /usr/include/cudnn* /workspace/cudnn_headers/include && \
                                                                                                           cp /usr/include/x86_64-linux-gnu/cudnn* /workspace/cudnn_headers/include_v8"

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker build --no-cache -t ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME} -f Dockerfile .

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME}