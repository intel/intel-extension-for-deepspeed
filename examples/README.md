## Recipes for Megatron-DeepSpeed
This folder contains recipes to run models of [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

To run any of the examples in this folder, please go to the base directory of Megatron-DeepSpeed and run as follows

```bash <path-to-this-repo>/examples/run3.6b.sh```

### Prepare dataset

To run recipes under Megatron-DeepSpeed, please setup your own dataset or use download scripts prepared by Megatron-DeepSpeed.

### Basic usage

For basic usage, we have provided 3 running recipes for 3.6 billion parameters, 20 billion parameters and 175 billion parameters training:

* 3.6b:     ```bash <path-to-this-repo>/examples/run3.6b.sh```
* 20b:      ```bash <path-to-this-repo>/examples/run20b.sh```
* 175b:     ```bash <path-to-this-repo>/examples/run175b.sh```

## Run with Huggingface
Intel-extension-for-deepspeed also works with [Huggingface Transformers](https://github.com/huggingface/transformers) and is able to do fine-tuning/inference tasks.

Install huggingface Transformers:
```bash
cd <path-to-this-repo>/examples
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
```

To run translation task with t5-small model on single gpu:
```bash
cd transformers
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --bf16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

To deploy on 8 gpus doing fine-tuing with Llama-2-7b model:
```bash
cd transformers
deepspeed --num_gpus=8 examples/pytorch/language-modeling/run_clm.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path meta-llama/Llama-2-7b-hf \
--dataset_name wikitext \
--dataset_config_name wikitext-2-raw-v1 \
--dataloader_num_workers 0 \
--per_device_train_batch_size 1 \
--warmup_steps 10 \
--max_steps 50 \
--bf16 \
--do_train \
--output_dir /tmp/test-clm \
--overwrite_output_dir
```

For detailed usage with huggingface/transformers, please check [transformers document](https://huggingface.co/docs/transformers/en/deepspeed).
