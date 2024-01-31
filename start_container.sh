#!/bin/bash

CONTAINER_NAME="analyzer"
IMAGE_NAME="u118nn8u2204"

# 檢查容器是否存在
if docker inspect $CONTAINER_NAME > /dev/null 2>&1; then
    # 容器存在
    echo "Container $CONTAINER_NAME already exists."
    docker start $CONTAINER_NAME
    docker exec -it $CONTAINER_NAME /bin/bash   
else
    # 容器不存在，創建新容器
    echo "Container $CONTAINER_NAME doesn't exist."
    docker run --gpus all -it --name $CONTAINER_NAME -v .:/app $IMAGE_NAME
fi


