#!/bin/bash

DTYPE=${DTYPE:-float16}
MODEL_PATH=${MODEL_PATH:-/home/username/model_path}
MODEL_NAME=${MODEL_NAME:-llama2-70b}
OUTPUT_DIR=logs/${MODEL_NAME}_`date +%m%d%H%M%S`_${HOSTNAME}
mkdir -p $OUTPUT_DIR

# Hostfile path
hostfile_deepspeed=$LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_deepspeed
hostfile_mpich=$LLM_DK_DIR/intel-extension-for-deepspeed/examples/hostfile_mpich

# launcher setting
LAUNCHER=${LAUNCHER:-MPICH}
if [[ $LAUNCHER == "deepspeed" ]]; then
    launcher=""
else
    launcher="--force_multi --hostfile $hostfile_deepspeed --launcher=${LAUNCHER} --launcher_args='-hostfile ${hostfile_mpich}'"
fi

CCL=${CCL:-ccl}

run_cmd="
    deepspeed $launcher run_generation_with_deepspeed.py \
    --device xpu \
    --ipex \
    --dtype $DTYPE \
    --input-tokens 1024 \
    --max-new-tokens 128 \
    --num-beam 1 \
    --batch-size 1 \
    --token-latency \
    --benchmark \
    -m $MODEL_PATH \
    --sub-model-name $MODEL_NAME\
    |& tee $OUTPUT_DIR/output.log
    "

echo ${run_cmd}
eval ${run_cmd}
set +x
