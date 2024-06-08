# clean
rm -fr ./output/*
rm -fr ./examples_deepspeed/ds_config_gpt*

bash examples_deepspeed/ds_pretrain_gpt.sh 1
bash examples_deepspeed/ds_pretrain_gpt.sh 2
bash examples_deepspeed/ds_pretrain_gpt.sh 3
bash examples_deepspeed/ds_pretrain_gpt.sh 4
bash examples_deepspeed/ds_pretrain_gpt.sh 5
bash examples_deepspeed/ds_pretrain_gpt.sh 6
