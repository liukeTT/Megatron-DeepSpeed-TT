wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
xz -d oscar-1GB.jsonl.xz

mkdir -p ${HOME}/llm-datasets/
mv oscar-1GB.jsonl ${HOME}/llm-datasets/
