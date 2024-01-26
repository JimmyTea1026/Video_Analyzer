#!/bin/bash

CONTAINER_NAME="video_analyzer"
IMAGE_NAME="c121nn8u2004"

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


