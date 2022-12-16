Megatron is a large, powerful transformer. This repo is for ongoing research on training large, powerful transformer language models at scale. Currently, we support multicards training of [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

The codebase is capable of efficiently training a 30-layer, 3.6 Billion Parameter GPT2 Language Model across Intel GPUs.


# Setup
We officially support python3.9.7 and we highly recommend to install an [Anaconda](https://www.anaconda.com/distribution/#download-section) environment.

## Prerequisites
To use this repo please install the specified versions of dependent software. You will need:
- Python 3.9.7 or later.
- Intel GPU driver for AI/compute workload

## Install Dependencies
Install Framework Dependency:
- [Intel® Extension for PyTorch\*](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-master) with XPU support.
- [Torch-ccl](https://github.com/intel/torch-ccl) with XPU support.

Install DeepSpeed Dependency:
- [DeepSpeed with XPU support](https://github.com/microsoft/DeepSpeed/pull/2221)
- [Intel® Extension for DeepSpeed\*](https://github.com/intel/intel-extension-for-deepspeed)


Create a virtual environment by conda and install dependent python packages:
```
conda create --name gpt2_env python=3.9.7
conda activate gpt2_env
pip install -r requirements.txt
```

# Usage
We've provided a script, gpt-3.6b.sh, for pretrain GPT2.

## GPT2 Pretraining
`bash scripts/gpt-3.6b.sh`

This script launches gpt2 pretraining that is verified on Intel GPUs.


```
python pretrain_gpt2.py \
       --model-parallel-size 1 \
       --num-layers 30 \
       --hidden-size 3072 \
       --num-attention-heads 32 \
       --batch-size 8 \
       --seq-length 2048 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --resume-dataloader \
       --train-data c4/en \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
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
       --bf16
```

## Datasets
We do not host any datasets for GPT2 training. However, we detail the collection so that our results can be reproduced. 

### Prepare c4/en Training Data
We use c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4). First, make sure you have [Git Large File Storage](https://git-lfs.github.com/) installed.

```
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git-lfs pull --include "en/*"
```

This will download c4/en about 305GB to your local device. Then you can run the following commands to get a train json named c4-train.json.
```
mkdir -p data/c4/en/
cat c4/en/c4-train* > data/c4/en/c4-train.json.gz
pushd data/c4/en
gzip -d c4-train.json.gz
popd
cat c4/en/c4-validation.0000* > data/c4/en/c4-validation.json.gz
```
If you don't need all c4/en datasets, you can run the following commands to merge 1024 original json.gz files into 8 json.gz files.
```
cd <path to c4>

mkdir -p softlinks
for shard in {0..7}; do
  start=$((shard * 128))
  end=$((shard * 128 + 127))
  mkdir -p softlinks/en_$shard
  for ind in $(seq -f "%05g" $start $end); do
    ln -s ../../en/c4-train.${ind}-of-01024.json.gz softlinks/en_${shard}/c4-train.${ind}-of-01024.json.gz
  done
done
mkdir -p en_merge
for shard in {0..7}; do 
  cat softlinks/en_${shard}/*gz > en_merge/c4-train.en_${shard}.json.gz 
done
```

If your system is memory limited we also recommend to run pretraining with the `--lazy-loader` argument as we've done. After preprocessing the dataset once, this will allow the dataset to be lazily loaded from disk, as opposed to storing it in memory.


### Aliasing datasets with corpora.py
We recommend aliasing datasets with human readable names (eg. `--train-data wikipedia`). This helps avoid forgetting arguments when submitting jobs, and allows one to combine datasets that would otherwise require different commandline options/data structures.

Examples of how to create these dataset objects can be found in [`./data_utils/corpora.py`](./data_utils/corpora.py). We recommend that the objects inherit from or adhere to the interface laid out by `torch.utils.data.Dataset` objects.

Any created datasets should be then added to the `NAMED_CORPORA` dictionary object in [`./data_utils/corpora.py`](./data_utils/corpora.py). At runtime one can specify one or more corpora from the commandline with `--train-data corpus1 corpus2 corpus3`, `--valid-data corpus1 corpus2 corpus3`, or `--test-data ...`.


### Partitioning datasets into Train/Val/Test
We support multiple ways to partition corpora into train/val/test splits. By specifying a `--split 95,5` commandline argument, the corpora specified by `--train-data` will have it's documents split proportionally into a 95%, 5% train/val split. The split is performed lazily on the fly and is efficient and deterministic from run to run given the same `--seed`. Note that if `--valid-data` or `--test-data` is specified then the train data will still be split accordingly, but `--valid-data`/`--test-data` will still be used as the validation/test source.

We do realize that this method, while effective, introduces noise into the development process, since different seeds will change the dataset and outcome. To have fixed training/validation/test sets across all your runs please utilize our script [`./scripts/split_json.py`](./scripts/split_json.py)

