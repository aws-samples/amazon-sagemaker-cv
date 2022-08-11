CONTAINER_NAME=mlperf
IMAGE_NAME=920076894685.dkr.ecr.us-east-1.amazonaws.com/jbsnyder:pytorch-mrcnn
docker run -it -d --rm --gpus all --name ${CONTAINER_NAME} \
    --net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
    --ulimit=stack=67108864 --ulimit=memlock=-1 \
    -w /opt/ml/code \
    -v /home/ubuntu/data:/opt/ml/input/data \
    -v /home/ubuntu/lilian:/opt/ml/code ${IMAGE_NAME}
