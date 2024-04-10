# docker build -t megatron -f ./Dockerfile .
# docker tag megatron:latest keliu354/megatron:v6
# docker push keliu354/megatron:v6
# docker system prune --volumes

docker_img="megatron:latest"

home_dir_0="${HOME}/Megatron-DeepSpeed-TT"
docker_dir_0="/workspace/Megatron-DeepSpeed"

home_dir_1="${HOME}/llm-datasets"
docker_dir_1="/workspace/datasets"

home_dir_2="${HOME}/llm-models"
docker_dir_2="/workspace/models"

work_dir=${docker_dir_0}

docker run --gpus all --rm -it -P \
    --cap-add SYS_NICE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${home_dir_0}:${docker_dir_0} \
    -v ${home_dir_1}:${docker_dir_1} \
    -v ${home_dir_2}:${docker_dir_2} \
    -w /${work_dir} \
    ${docker_img} \
    bash
