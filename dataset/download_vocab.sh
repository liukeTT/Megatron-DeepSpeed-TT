#wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
#wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

mkdir -p ${HOME}/llm-datasets/
mv gpt2-vocab.json ${HOME}/llm-datasets/
mv gpt2-merges.txt ${HOME}/llm-datasets/