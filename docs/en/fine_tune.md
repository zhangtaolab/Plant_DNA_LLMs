To fine-tune the plant DNA LLMs, please first download the desired models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local. You can use `git clone` (which may require `git-lfs` to be installed) to retrieve the model or directly download the model from the website.

In the activated `llms` python environment, use the `model_finetune.py` script to fine-tune a model for downstream task.  

Our script accepts `.csv` format data (separated by `,`) as input, when preparing the training data, please make sure the data contain a header and at least these two columns:
```csv
sequence,label
```
Where `sequence` is the input sequence, and `label` is the corresponding label for the sequence.

We also provide several plant genomic datasets for fine-tuning on the [HuggingFace](https://huggingface.co/zhangtaolab) and [ModelScope](https://www.modelscope.cn/organization/zhangtaolab).

* Here is the [pretrain models list](resources/pretrain_models.md)

We use Plant DNAGPT model as example to predict active core promoter.

First download a pretrain model and corresponding dataset from HuggingFace or ModelScope:

```bash
# prepare a work directory
mkdir LLM_finetune
cd LLM_finetune
# download pretrain model
git clone https://modelscope.cn/models/zhangtaolab/plant-dnagpt-BPE
# download train dataset
git clone https://modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters
```

After preparing the model and dataset, using the following script to finetune model (here is a promoter prediction example)

```bash
python model_finetune.py \
    --model_name_or_path plant-dnagpt-BPE \
    --train_data plant-multi-species-core-promoters/train.csv \
    --test_data plant-multi-species-core-promoters/test.csv \
    --eval_data plant-multi-species-core-promoters/dev.csv \
    --train_task classification \
    --labels 'Not promoter;Core promoter' \
    --run_name plant_dnagpt_BPE_promoter \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --load_best_model_at_end \
    --metric_for_best_model 'f1' \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir plant-dnagpt-BPE-promoter
```

In this script:  
1. `--model_name_or_path`: Path to the foundation model you downloaded
2. `--train_data`: Path to the train dataset
3. `--test_data`: Path to the test dataset, omit it if no test data available
4. `--dev_data`: Path to the validation dataset, omit it if no validation data available
5. `--train_task`: Determine the task type, should be classification, multi-classification or regression
6. `--labels`: Set the labels for classification task, separated by `;`
7. `--run_name`: Name of the fine-tuned model
8. `--per_device_train_batch_size`: Batch size for training model
9. `--per_device_eval_batch_size`: Batch size for evaluating model
10. `--learning_rate`: Learning rate for training model
11. `--num_train_epochs`: Epoch for training model (also you can train model with steps, then you should change the strategies for save, logging and evaluation)
12. `--load_best_model_at_end`: Whether to load the model with the best performance on the evaluated data, default is `True`
13. `--metric_for_best_model`: Use which metric to determine the best model, default is `loss`, can be `accuracy`, `precison`, `recall`, `f1` or `matthews_correlation` for classification task, and `r2` or `spearmanr` for regression task
14. `--save_strategy`: Strategy for saving model, can be `epoch` or `steps`
15. `--logging_strategy`: Strategy for logging training information, can be `epoch` or `steps`
16. `--evaluation_strategy`: Strategy for evaluating model, can be `epoch` or `steps`
17. `--output_dir`: Where to save the fine-tuned model

Detailed descriptions of the arguments can be referred [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

Finally, wait for the progress bar completed, and the fine-tuned model will be saved in the `plant-dnagpt-BPE-promoter` directory. In this directory, there will be a checkpoint directory, a runs directory, and a saved fine-tuning model.