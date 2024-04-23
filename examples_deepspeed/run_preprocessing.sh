dataset_dir="/workspace/datasets"
corpus_f="${dataset_dir}/oscar-1GB.jsonl"
vocab_f="${dataset_dir}/gpt2-vocab.json"
merge_f="${dataset_dir}/gpt2-merges.txt"
output_prefix="oscar-gpt2"
n_workers=64

#export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python tools/preprocess_data.py \
       --input ${corpus_f} \
       --output-prefix ${output_prefix} \
       --vocab-file ${vocab_f} \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ${merge_f} \
       --append-eod \
       --workers $n_workers

mv oscar-gpt2_text_document.bin ${dataset_dir}/
mv oscar-gpt2_text_document.idx ${dataset_dir}/
