# launch training
# these are the DGX-A100 settings 
# but with instance specific optimization removed
# should train to convergence on a p4d in a little under 1 hour

REPO=${1-ecr-pt-repo}
TAG=pytorch-mrcnn
REGION=${2-us-east-1}
AWS_ACCOUNT=`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`

CONTAINER_NAME=mlperf_training
IMAGE_NAME=${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
BASE_LR=0.06
MAX_ITER=20000
WARMUP_FACTOR=0.000096
WARMUP_ITERS=625
STEPS="\"(12000,16000)\""
TRAIN_IMS_PER_BATCH=48
TEST_IMS_PER_BATCH=24
FPN_POST_NMS_TOP_N_TRAIN=6000
NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=20
NPROC_PER_NODE=4
ENABLE_DALI=False
USE_CUDA_GRAPH=True
CACHE_EVAL_IMAGES=True
EVAL_SEGM_NUMPROCS=10
EVAL_MASK_VIRTUAL_PASTE=True
INCLUDE_RPN_HEAD=True
PRECOMPUTE_RPN_CONSTANT_TENSORS=True
DATALOADER_NUM_WORKERS=1
HYBRID_LOADER=True
FPN_POST_NMS_TOP_N_PER_IMAGE=False
PATHS_CATALOG=/workspace/src/paths_catalog.py

docker run -it --rm --gpus all --name ${CONTAINER_NAME} \
	--net=host --uts=host --ipc=host --security-opt=seccomp=unconfined \
	--ulimit=stack=67108864 --ulimit=memlock=-1 \
	-w /workspace/src \
	-v /home/ec2-user/SageMaker/data/mlperf:/opt/ml/input/data \
    -v /home/ec2-user/SageMaker/mlperf_mrcnn/src:/workspace/src ${IMAGE_NAME} /bin/bash -c "cd /workspace/src && ./run.sh"
