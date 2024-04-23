cd ${HOME}
sudo rm -fr ${HOME}/Megatron-LM

git clone git@github.com:NVIDIA/Megatron-LM.git
cp ${HOME}/Megatron-DeepSpeed-TT/examples_core/mcore_train.py ${HOME}/Megatron-LM/