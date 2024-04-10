# clean old checkpoint
checkpoint_path="/tmp/checkpoint"
rm -fr $checkpoint_path

# clean prof trace
prof_path="./kineto"
rm -fr $prof_path
mkdir -p $prof_path

prof_path="./et"
rm -fr $prof_path
mkdir -p $prof_path

# config train
num_train_epochs=1
train_batch_size=2
eval_batch_size=2
max_train_samples=32
max_eval_samples=8

_args_train=" --model_name_or_path gpt2"\
" --dataset_name wikitext"\
" --dataset_config_name wikitext-2-raw-v1"\
" --num_train_epochs "${num_train_epochs}\
" --per_device_train_batch_size "${train_batch_size}\
" --per_device_eval_batch_size "${eval_batch_size}\
" --max_train_samples "${max_train_samples}\
" --max_eval_samples "${max_eval_samples}\
" --do_train"\
" --do_eval"\
" --output_dir "${checkpoint_path}

# python run
python lm_train.py $_args_train

# deepspeed run

ds_config="ds_config.json"
#ds_config="ds_config_offload.json"
_args_deepspeed=" --deepspeed "${ds_config}

#deepspeed lm_train.py $_args_train $_args_deepspeed
