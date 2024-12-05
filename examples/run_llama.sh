# please make sure the content of hostfile for single node is localhost
export WORLD_SIZE=${WORLD_SIZE:-48}
export MICRO_BATCH=${MICRO_BATCH:-1}
export NLAYERS=${NLAYERS:-32}
export HIDDEN=${HIDDEN:-4096}
export HEADS=${HEADS:-32}
export SEQ=${SEQ:-2048}
export NUM_KV_HEADS=${NUM_KV_HEADS:-32}
export FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-11008}
export TRAIN_ITER=${TRAIN_ITER:-50}
export ZERO_STAGE=${ZERO_STAGE:-3}
export DTYPE=${DTYPE:-bf16}
export TP=${TP:-1}
export PP=${PP:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH * $GRAD_ACC_STEPS / $TP / $PP ))

bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/gpt.sh --no-query-key-layer-scaling \
--use-rotary-position-embeddings --untie-embeddings-and-output-weights --swiglu --disable-bias-linear \
--normalization rmsnorm --attention-dropout 0 --hidden-dropout 0 --use-flash-attn-builder \
--ffn-hidden-size $FFN_HIDDEN_SIZE --num-key-value-heads $NUM_KV_HEADS $@
