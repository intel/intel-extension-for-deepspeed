echo "!!!please makes sure the content of hostfile for single node is localhost"
export WORLD_SIZE=12
export MICRO_BATCH=8
export NLAYERS=30
export HIDDEN=3072
export HEADS=32
export SEQ=2048
export TRAIN_ITER=50
export ZERO_STAGE=2
export TP=1
export PP=1
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH / $TP / $PP ))

export DS_CONFIG=${PWD}/"ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_bf16.json"

bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/generate_config.sh
bash $LLM_DK_DIR/intel-extension-for-deepspeed/examples/gpt.sh --no-query-key-layer-scaling