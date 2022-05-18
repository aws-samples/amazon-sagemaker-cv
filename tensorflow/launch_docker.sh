IMAGE_NAME=tf_28
REGION=us-east-1
AWS_ACCOUNT=763104351884
REPO=tensorflow-training
TAG=2.8.0-gpu-py39-cu112-ubuntu20.04-sagemaker

docker run --rm -it -d --gpus all --name ${IMAGE_NAME} --net=host --uts=host --ipc=host \
	--ulimit=stack=67108864 --ulimit=memlock=-1 --security-opt=seccomp=unconfined \
	-v ~/data/:/workspace/data/ \
	-v ~/amazon-sagemaker-cv/:/workspace/smcv/ \
	${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
