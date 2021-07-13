REPO=$1
REGION=${2-us-west-2}
AWS_ACCOUNT=${3-`aws sts get-caller-identity --region ${REGION} --endpoint-url https://sts.${REGION}.amazonaws.com --query Account --output text`}
IMAGE_NAME=${4-tf-2.5-cudnn-8.2}

docker build -t ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME} -f Dockerfile .

aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com

docker push ${AWS_ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}:${IMAGE_NAME}
