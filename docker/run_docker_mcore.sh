# docker build -t megatron-core -f ./Dockerfile_mcore .
# docker system prune --volumes

docker_img="megatron-core:latest"

home_dir_0="${HOME}/Megatron-DeepSpeed-TT"
docker_dir_0="/workspace/Megatron-DeepSpeed"

home_dir_1="${HOME}/Megatron-LM"
docker_dir_1="/workspace/Megatron-LM"

home_dir_data="${HOME}/llm-datasets"
docker_dir_data="/workspace/datasets"

home_dir_model="${HOME}/llm-models"
docker_dir_model="/workspace/models"

work_dir=${docker_dir_1}

docker run --gpus all --rm -it -P \
    --cap-add SYS_NICE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${home_dir_0}:${docker_dir_0} \
    -v ${home_dir_1}:${docker_dir_1} \
    -v ${home_dir_data}:${docker_dir_data} \
    -v ${home_dir_model}:${docker_dir_model} \
    -w /${work_dir} \
    ${docker_img} \
    bash
