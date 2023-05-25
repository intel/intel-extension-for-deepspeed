#!/bin/bash

VOCAB_FILE=dataset/gpt2-vocab.json
MERGE_FILE=dataset/gpt2-merges.txt
DATA_PATH=dataset/BookCorpusDataset_text_document
DS_CONFIG=${DS_CONFIG:-${PWD}/"ds_stage_${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_bf16.json"}

# Hostfile path
hostfile_deepspeed=$LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_deepspeed
hostfile_mpich=$LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_mpich

# Disabling tensor/pipeline parallelism
TP=${TP:-1}
PP=${PP:-1}

# Model: default 3.6b
NLAYERS=${NLAYERS:-30}
HIDDEN=${HIDDEN:-3072}
HEADS=${HEADS:-32}
SEQ=${SEQ:-2048}
TRAIN_ITER=${TRAIN_ITER:-50}


WORLD_SIZE=${WORLD_SIZE:-12}
MICRO_BATCH=${MICRO_BATCH:-8}
GLOBAL_BATCH=${GLOBAL_BATCH:-96}

ZERO_STAGE=${ZERO_STAGE:-2}
OUTPUT_DIR=logs/ds_stage${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_`date +%m%d%H%M%S`
mkdir -p $OUTPUT_DIR
echo "!!!Please see logs at ${OUTPUT_DIR}"

ds_args=" "
ds_args=" --deepspeed ${ds_args}"
if [ $PP == 1 ]; then
   ds_args=" --no-pipeline-parallel ${ds_args}" 
fi
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
# we are now using activation checkpoint provided by megatron, see below.
# ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

# take custom args
custom_args=" $@"

# launcher setting
launcher=MPICH

run_cmd="
    deepspeed --force_multi --hostfile $hostfile_deepspeed --launcher=${launcher} --launcher_args='-hostfile ${hostfile_mpich}' pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --num-attention-heads $HEADS \
    --seq-length $SEQ \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $TRAIN_ITER \
    --lr 0.00015 \
    --lr-warmup-fraction .01 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 100 \
    --eval-interval 100 \
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --save-interval 500 \
    --split 100,0,0 \
    --bf16 \
    --checkpoint-activations \
    $ds_args \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --distributed-backend ccl \
    --num-workers 0 \
    $custom_args \
    | tee $OUTPUT_DIR/output.log
    "

echo ${run_cmd}
eval ${run_cmd}
set +x
