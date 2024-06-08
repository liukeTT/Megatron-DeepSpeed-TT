#!/bin/bash
DIR=`pwd`
parallelism=$1

###############################################################################
### GPT-125M
MODEL_SIZE="125M"
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTN_HEADS=12

# Config GPT to be MoE: 1 means dense GPT
EP_SIZE=1

###############################################################################
### Parallelism configs

## NUM_GPUS = DP x TP x PP
NUM_GPUS=4

## Data parallelism (DP)
## Tensor parallelism (TP): 1 means no TP
## Pipeline parallelism (PP): 1 means no PP

## BATCH_SIZE: batch size per GPU
## GLOBAL_BATCH_SIZE

if [ "${parallelism}" = 1 ]; then
        DP_SIZE=4
        TP_SIZE=1
        PP_SIZE=1
        BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=256
elif [ "${parallelism}" = 2 ]; then
        DP_SIZE=1
        TP_SIZE=4
        PP_SIZE=1
        BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=256
elif [ "${parallelism}" = 3 ]; then
        DP_SIZE=1
        TP_SIZE=1
        PP_SIZE=4
        BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=256
elif [ "${parallelism}" = 4 ]; then
        DP_SIZE=1
        TP_SIZE=2
        PP_SIZE=2
        BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=256
elif [ "${parallelism}" = 5 ]; then
        DP_SIZE=2
        TP_SIZE=1
        PP_SIZE=2
        BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=256
elif [ "${parallelism}" = 6 ]; then
        DP_SIZE=2
        TP_SIZE=2
        PP_SIZE=1
        BATCH_SIZE=4
        GLOBAL_BATCH_SIZE=256
else
        echo "No such parallelism"
        exit
fi

## Experts parallelism: MoE only, 1 means no EP
if [ $EP_SIZE -gt $NUM_GPUS ]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi


###############################################################################
### Training configs

## training samples
VOCAB_PATH="/workspace/llm-datasets/gpt2-vocab.json"
MERGE_PATH="/workspace/llm-datasets/gpt2-merges.txt"
DATA_BLEND="/workspace/llm-datasets/oscar-gpt2_text_document"

SEQ_LEN=1024
TRAIN_SAMPLES=1000
TRAIN_TOKENS=$(( ${TRAIN_SAMPLES} * ${SEQ_LEN} ))

## Another termination condition in minutes
EXIT_DURATION=10

## LR configs
WARMUP_TOKENS=$(( 10 * ${SEQ_LEN} ))
LR_DECAY_TOKENS=$(( 100 * ${SEQ_LEN} ))

LR=6.0e-4
MIN_LR=6.0e-5

###############################################################################
### Misc configs
LOG_INTERVAL=1
EVAL_ITERS=10
EVAL_INTERVAL=100
SAVE_INTERVAL=1000

## Standard deviation for weight initialization
INIT_STD=0.014
ACTIVATION_CHECKPOINT="false"

###############################################################################
### Output, prof and data configs
NAME="gpt-${MODEL_SIZE}-bs-${BATCH_SIZE}-dp-${DP_SIZE}-tp-${TP_SIZE}-pp-${PP_SIZE}"


# clean old logs
OUTPUT_BASEPATH=$DIR/output/$NAME
rm -fr ${OUTPUT_BASEPATH}

# prof path
PROF_PATH="${OUTPUT_BASEPATH}"
mkdir -p ${OUTPUT_BASEPATH}/et/
mkdir -p ${OUTPUT_BASEPATH}/kineto/

# profiler
PROF="true"
PROF_STACK="true"
PROF_STATR=3
PROF_STOP=3

###############################################################################
### args: Megatron

data_options=" \
        --vocab-file ${VOCAB_PATH} \
        --merge-file ${MERGE_PATH} \
        --data-path ${DATA_BLEND} \
        --data-impl mmap"
        
megatron_options=" \
        --override-opt_param-scheduler \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --init-method-std ${INIT_STD} \
        --lr-decay-tokens ${LR_DECAY_TOKENS} \
        --lr-warmup-tokens ${WARMUP_TOKENS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --rampup-batch-size 32 32 1953125 \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --train-tokens ${TRAIN_TOKENS} \
        --train-samples ${TRAIN_SAMPLES} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --split 98,2,0 \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --save-interval ${SAVE_INTERVAL} \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --hysteresis 2 \
        --num-workers 0 \
        --fp16"

# 

if [ $EP_SIZE -gt 1 ]; then
        megatron_options="${megatron_options} \
        --no-pipeline-parallel \
        --create-moe-param-group"
fi

if [ "${PROF}" = "true" ];
then
        megatron_options="${megatron_options} \
        --profile \
        --profile-step-start ${PROF_STATR} \
        --profile-step-end ${PROF_STOP} \
        --profile-trace-path ${PROF_PATH} "
fi

if [ "${PROF_STACK}" = "true" ]; then
        megatron_options="${megatron_options} \
        --with-stack"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ];
then
        megatron_options="${megatron_options} \
        --checkpoint-activations"
fi



###############################################################################
### args: DeepSpeed

ZERO_STAGE=0
OFFLOAD="false"

CONFIG_FP16_ENABLED="true"
CONFIG_BF16_ENABLED="false"

if [ "${OFFLOAD}" = "false" ]; then
        template_json="examples_deepspeed/ds_config_TEMPLATE.json"
        config_json="examples_deepspeed/ds_config_${NAME}.json"
fi

if [ "${OFFLOAD}" = "true" ]; then
        template_json="examples_deepspeed/ds_config_offload_TEMPLATE.json"
        config_json="examples_deepspeed/ds_config_offload_${NAME}.json"
fi

sed "s/CONFIG_BATCH_SIZE/${GLOBAL_BATCH_SIZE}/" ${template_json} \
        | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
        | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
        | sed "s/ZERO_STAGE/${ZERO_STAGE}/" \
        | sed "s/PRESCALE_GRAD/true/" \
        | sed "s/CONFIG_FP16_ENABLED/${CONFIG_FP16_ENABLED}/" \
        | sed "s/CONFIG_BF16_ENABLED/${CONFIG_BF16_ENABLED}/" \
        > ${config_json}

deepspeed_options=" \
        --deepspeed \
	--deepspeed_config ${config_json}"

if [ "${OFFLOAD}" = "true" ]; then
        deepspeed_options="${deepspeed_options} \
        --cpu-optimizer"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
        deepspeed_options="${deepspeed_options} \
        --deepspeed-activation-checkpointing"
fi

run_cmd="deepspeed pretrain_gpt.py ${data_options} ${megatron_options} ${deepspeed_options} > ${OUTPUT_BASEPATH}/${NAME}.log"
echo ${run_cmd}
eval ${run_cmd}
set +x