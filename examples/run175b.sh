echo "!!!please use generate_hostfile.sh to set hostfile for 18 nodes before training"
export WORLD_SIZE=216
export MICRO_BATCH=1
export NLAYERS=96
export HIDDEN=12288
export HEADS=96
export SEQ=2048
export TRAIN_ITER=20
export ZERO_STAGE=3
export TP=1
export PP=1
export GLOBAL_BATCH=$(( $WORLD_SIZE * $MICRO_BATCH / $TP / $PP ))

export DS_CONFIG=${PWD}/"ds_stage${ZERO_STAGE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}_pp${PP}_tp${TP}_bf16.json"

bash ${intel-extension-for-deepspeed}/examples/generate_config.sh
bash ${intel-extension-for-deepspeed}/examples/gpt.sh