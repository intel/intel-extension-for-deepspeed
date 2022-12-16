#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=12


script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="$script_dir/ds_zero2_config_bf16.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 30 \
       --hidden-size 3072 \
       --num-attention-heads 32 \
       --batch-size 8 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 10 \
       --resume-dataloader \
       --train-data c4/en \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend ccl \
       --lr 0.00015 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing \
       --bf16 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"

ds_args=""
gpt2_args=""

for i in "$@"; do
  if [[ $i =~ "--oneprof_args" ]]; then
    ds_args="$ds_args $i"
  elif [[ $i =~ "--onetrace_args" ]]; then
    ds_args="$ds_args $i"
  else
    gpt2_args="$gpt2_args $i"
  fi
done

run_cmd="deepspeed $ds_args --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py ${gpt_options} $gpt2_args"
echo ${run_cmd}
eval ${run_cmd}

set +x
