echo "!!!please use generate_hostfile.sh to set hostfile for 18 nodes before training"
export WORLD_SIZE=${WORLD_SIZE:-216}
export MICRO_BATCH=${MICRO_BATCH:-1}
export NLAYERS=${NLAYERS:-96}
export HIDDEN=${HIDDEN:-12288}
export HEADS=${HEADS:-96}
export SEQ=${SEQ:-2048}
export TRAIN_ITER=${TRAIN_ITER:-20}
export ZERO_STAGE=${ZERO_STAGE:-3}
export DTYPE=${DTYPE:-bf16}
export TP=${TP:-1}
export PP=${PP:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))

bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/gpt.sh $@
