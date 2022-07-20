#!/bin/bash

# Get region of the current instance
ZONE=`curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone`
REGION="`echo \"$ZONE\" | sed 's/[a-z]$//'`"

# ECR repo and image tag
REPO=${1-sm-images}
TAG=${2-pytorch-yolo}

# Get AWS account number associated with instance
AWS_ACCOUNT=`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`

# Build docker image
docker build -t ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG} -f Dockerfile .

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

# Push image to ECR
docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${TAG}