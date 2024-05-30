# docker build -t megatron-core -f ./Dockerfile_mcore .
# docker system prune --volumes

docker_img="megatron-core:latest"

home_dir_0="${HOME}/nvbandwidth"
docker_dir_0="/workspace/nvbandwidth"

home_dir_0="${HOME}/BabelStream"
docker_dir_0="/workspace/BabelStream"

work_dir=${docker_dir_0}

docker run --gpus all --rm -it -P \
    --cap-add SYS_NICE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${home_dir_0}:${docker_dir_0} \
    -w /${work_dir} \
    ${docker_img} \
    bash
