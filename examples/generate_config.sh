#!/bin/bash

if [ -z $ZERO_STAGE ]; then
  echo "Please set your ZERO_STAGE before generateing config file!"
  return 0
fi

if [ -z $PP ]; then
  echo "Please set your PP_SIZE before generateing config file!"
  return 0
fi

if [ $ZERO_STAGE == 3 ]; then
  cat <<EOT > $DS_CONFIG
  {
    "train_batch_size": $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "steps_per_print": 1,
    "gradient_accumulation_steps": 1,
    "optimizer": {
      "type": "Adam",
      "params": {
        "lr": 0.00015,
        "weight_decay": 1e-2
      }
    },
    "zero_optimization": {
      "stage": 3,
      "reduce_scatter": false,
      "stage3_max_live_parameters": 3e9,
      "stage3_max_reuse_distance": 3e9,
      "stage3_param_persistence_threshold": 1e5,
      "stage3_prefetch_bucket_size": 5e7,
      "contiguous_gradients": true,
      "overlap_comm": true,
      "reduce_bucket_size": 90000000,
      "sub_group_size": 1e9,
      "offload_optimizer": {
        "device": "none",
        "buffer_count": 4,
        "pipeline_read": false,
        "pipeline_write": false,
        "pin_memory": true
      }
    },
    "zero_allow_untested_optimizer": true,
    "communication_data_type": "bfp16",
    "gradient_clipping": 1.0,
    "fp16": {
      "enabled": false,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "bfloat16": {
      "enabled": true,
      "loss_scale": 1.0
    },
    "activation_checkpointing": {
      "partition_activations": true,
      "contiguous_memory_optimization": false
    },
    "wall_clock_breakdown": false,
    "flops_profiler": {
      "enabled": false,
      "profile_step": 45,
      "module_depth": -1,
      "top_modules": 1,
      "detailed": true,
      "output_file": null
    }
  }
EOT

elif [ $ZERO_STAGE == 2 ]; then
  cat <<EOT > $DS_CONFIG
  {
    "train_batch_size": $GLOBAL_BATCH,
    "train_micro_batch_size_per_gpu": $MICRO_BATCH,
    "steps_per_print": 1,
    "gradient_accumulation_steps": 1,
    "optimizer": {
      "type": "Adam",
      "params": {
        "lr": 0.00015,
        "weight_decay": 1e-2
      }
    },
    "zero_optimization": {
      "stage": $ZERO_STAGE
    },
    "zero_allow_untested_optimizer": true,
    "communication_data_type": "bfp16",
    "gradient_clipping": 1.0,
    "fp16": {
      "enabled": false,
      "loss_scale": 0,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1
    },
    "bfloat16": {
      "enabled": true,
      "loss_scale": 1.0
    },
    "activation_checkpointing": {
      "partition_activations": true,
      "contiguous_memory_optimization": false
    },
    "wall_clock_breakdown": false,
    "flops_profiler": {
      "enabled": false,
      "profile_step": 45,
      "module_depth": -1,
      "top_modules": 1,
      "detailed": true,
      "output_file": null
    }
  }
EOT

elif [ $ZERO_STAGE == 1 ]; then
  if [ $PP > 1 ]; then
    cat <<EOT > $DS_CONFIG
    {
      "train_batch_size": $GLOBAL_BATCH,
      "train_micro_batch_size_per_gpu": $MICRO_BATCH,
      "steps_per_print": 1,
      "gradient_accumulation_steps": 1,
      "optimizer": {
        "type": "Adam",
        "params": {
          "lr": 0.00015,
          "weight_decay": 1e-2
        }
      },
      "zero_optimization": {
        "stage": $ZERO_STAGE
      },
      "zero_allow_untested_optimizer": true,
      "communication_data_type": "bfp16",
      "gradient_clipping": 1.0,
      "fp16": {
        "enabled": false,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
      },
      "bfloat16": {
        "enabled": true,
        "loss_scale": 1.0
      },
      "activation_checkpointing": {
        "partition_activations": true,
        "contiguous_memory_optimization": false
      },
      "wall_clock_breakdown": false,
      "flops_profiler": {
        "enabled": false,
        "profile_step": 45,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": true,
        "output_file": null
      },
      "data_types": {
        "grad_accum_dtype": "fp32"
      },
      "comms_logger": {
        "enabled": true,
        "verbose": false,
        "prof_all": true,
        "debug": false
      }
    }
EOT
  else
    echo "please add the config for zero_stage 1 without pipeline-parallelism"
  fi  

else
  echo "Please add the correct config set!!!"
fi
