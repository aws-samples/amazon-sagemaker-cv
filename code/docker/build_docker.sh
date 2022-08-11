REPO=${1-ecr-pt-repo}
TAG=pytorch-mrcnn
REGION=${2-us-east-1}

# Grab current AWS account from sts
AWS_ACCOUNT=`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`

# Build Docker image based on Nvidia PT image
docker build -t ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG} -f Dockerfile .

# Log in to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# Push image to ECR
docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}
