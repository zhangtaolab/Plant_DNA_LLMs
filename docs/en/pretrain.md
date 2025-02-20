To pretrain our DNA models from scratch, please first download the desired pretrained models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local. You can use `git clone` (which may require `git-lfs` to be installed) to retrieve the model or directly download the model from the website.

In the activated `llms` python environment, use the `model_pretrain_from_scratch.py` script to pretrain a model use your own dataset. 

Before training the model, please prepare a training dataset that contains the splited sequences (length shorted than 2000 bp) from the target genomes. (Detailed information can be found in the supplemental notes out our [manuscript](https://www.cell.com/molecular-plant/fulltext/S1674-2052(24)00390-3))

The dataset file should look like [this](https://huggingface.co/datasets/zhangtaolab/plant-reference-genomes).

* Here is the [pretrain models list](docs/en/resources/pretrain_models.md)

We use DNAGPT model as an example to perform the pretrain.

```bash
# prepare a output directory
mkdir pretrain
# download pretrain model
git clone https://huggingface.co/zhangtaolab/plant-dnagpt-BPE models/plant-dnagpt-BPE
# prepare your own dataset for pretraining, data can be stored at the data directory
# example: data/pretrain_data.txt
```

* Note: If downloading from huggingface encounters network error, please try to download model from ModelScope or change to the accelerate mirror before downloading.
```bash
# Download with git
git clone https://hf-mirror.com/[organization_name/repo_name]
# Download with huggingface-cli
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download [organization_name/repo_name]
```

After preparing the model and dataset, using the following script to pretrain the model.

```bash
python model_pretrain_from_scratch.py \
    --model_name_or_path models/plant-dnagpt-BPE \
    --train_data data/pretrain_data.txt \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 24 \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --bf16 \
    --logging_strategy steps \
    --logging_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --output_dir pretrain/dnagpt-BPE_updated
```

In this script:  
1. `--model_name_or_path`: Path to the foundation model you downloaded
2. `--train_data`: Path to the train dataset
3. `--per_device_train_batch_size`: Batch size for training model
4. `--gradient_accumulation_steps`: Number of updates steps to accumulate the gradients for, before performing a backward/update pass
5. `--num_train_epochs`: Epoch for training model (also you can train model with steps, then you should change the strategies for save, logging and evaluation)
6. `--learning_rate`: Learning rate for training model
7. `--warmup_ratio`: Ratio of total training steps used for a linear warmup from 0 to `learning_rate`
8. `--bf16`: Use bf16 precision for training
9. `--logging_strategy`: Strategy for logging training information, can be `epoch` or `steps`
10. `--logging_steps`: Steps for logging training information
11. `--save_strategy`: Strategy for saving model, can be `epoch` or `steps`
12. `--save_steps`: Steps for saving model checkpoints
13. `--output_dir`: Where to save the pretrained model

Detailed descriptions of the arguments can be referred [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

Finally, wait for the progress bar completed, and the pretrained model will be saved in the `pretrain/dnagpt-BPE_updated` directory. In this directory, there will be checkpoint directories, a runs directory, and a saved pretrained model.
