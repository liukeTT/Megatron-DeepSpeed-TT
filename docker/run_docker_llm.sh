# docker build -t megatron-core -f ./Dockerfile_mcore .
# docker system prune --volumes

docker_img="megatron-core:latest"

home_dir_0="${HOME}/llm.c"
docker_dir_0="/workspace/llm.c"

home_dir_1="${HOME}/llm.tt"
docker_dir_1="/workspace/llm.tt"

work_dir=${docker_dir_1}

docker run --gpus all --rm -it -P \
    --cap-add SYS_NICE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${home_dir_0}:${docker_dir_0} \
    -v ${home_dir_1}:${docker_dir_1} \
    -w /${work_dir} \
    ${docker_img} \
    bash
