echo "!!!please makes sure the content of hostfile for single node is localhost"
export WORLD_SIZE=${WORLD_SIZE:-12}
export MICRO_BATCH=${MICRO_BATCH:-8}
export NLAYERS=${NLAYERS:-30}
export HIDDEN=${HIDDEN:-3072}
export HEADS=${HEADS:-32}
export SEQ=${SEQ:-2048}
export TRAIN_ITER=${TRAIN_ITER:-50}
export ZERO_STAGE=${ZERO_STAGE:-2}
export DTYPE=${DTYPE:-bf16}
export TP=${TP:-1}
export PP=${PP:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))

bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/gpt.sh --no-query-key-layer-scaling $@
