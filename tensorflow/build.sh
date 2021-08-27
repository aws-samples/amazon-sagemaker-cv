REPO=$1
REGION=${2-us-west-2}
AWS_ACCOUNT=${3-`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`}
IMAGE_NAME=${4-smcv-tf-2.4}
DLC_ACCOUNT=763104351884
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${DLC_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker build -t ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME} -f Dockerfile .

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME}