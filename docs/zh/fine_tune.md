微调植物DNA大语言模型前，需要先从[ModelScope](https://www.modelscope.cn/organization/zhangtaolab)或[HuggingFace](https://huggingface.co/zhangtaolab)网站下载需要的模型。可以使用 `git clone` 命令行（确保 `git-lfs` 命令正确安装）下载模型，或者直接在网页点击下载将模型下载到本地。

在激活的 `llms` conda环境下，可以使用 `model_finetune.py` 脚本针对下游任务微调模型。

脚本接受 `.csv` 格式的输入文件（以逗号 `,` 分隔），当准备训练数据时，确保数据文件包含表头，且至少包含序列和标签这两列：
```csv
sequence,label
```

`sequence`对应的是输入序列， `label`代表输入序列对应的类别或得分。

用户可以使用自己的数据做模型微调，此外我们也提供了一系列的植物基因组数据集用于微调模型的训练。用户可以在[ModelScope](https://www.modelscope.cn/organization/zhangtaolab)和[HuggingFace](https://huggingface.co/zhangtaolab)网站自行下载使用。

* 预训练模型的列表可参考 [预训练模型列表](resources/pretrain_models.md)

这里我们以基于BPE tokenizer的Plant DNAGPT模型为例，微调预测植物活性启动子的大语言模型。

首先先从ModelScope或Huggingface下载微调模型和对应的数据集：

```bash
# 准备一个工作目录
mkdir LLM_finetune
cd LLM_finetune
# 下载预训练模型
git clone https://modelscope.cn/models/zhangtaolab/plant-dnagpt-BPE
# 下载训练数据集
git clone https://modelscope.cn/datasets/zhangtaolab/plant-multi-species-core-promoters
```

模型文件和训练数据准备好后，可以使用如下命令微调模型（这里以预测启动子任务为例）：

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

以上的命令中，不同参数的介绍如下：  
1. `--model_name_or_path`: 基础DNA模型的路径
2. `--train_data`: 待训练数据中训练集的路径
3. `--test_data`: 待训练数据中测试集的路径（如果没有可以忽略）
4. `--dev_data`: 待训练数据中验证集的路径（如果没有可以忽略）
5. `--train_task`: 任务类型，可以是`classification`（二分类）、`multi-classification`（多分类）和`regression`（回归）
6. `--labels`: 分类任务的标签，以 `;` 分隔
7. `--run_name`: 微调模型的名称
8. `--per_device_train_batch_size`: 模型训练时的batch size
9. `--per_device_eval_batch_size`: 模型评估时的batch size
10. `--learning_rate`: 学习率
11. `--num_train_epochs`: 模型训练时的epoch数 (如果希望以`steps`进行训练，使用`--num_train_steps`参数, 需要确保模型保存、日志和评估的方法也都改成`steps`)
12. `--load_best_model_at_end`: 是否在训练完成后保存最好的模型，默认是 `True`
13. `--metric_for_best_model`: 使用哪个指标评估最好的模型, 默认是 `loss`, 还可以是 `accuracy`, `precison`, `recall`, `f1` 或 `matthews_correlation`（分类任务）,  `r2` 或 `spearmanr`（回归任务）
14. `--save_strategy`: 模型保存的方法，可以是 `epoch` or `steps`
15. `--logging_strategy`: 模型训练日志输出的方法，可以是 `epoch` or `steps`
16. `--evaluation_strategy`: 模型评估的方法，可以是  `epoch` or `steps`
17. `--output_dir`: 微调模型的保存路径

更多关于参数的细节，可以参考[transformers官方文档](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)。

最后，等待进度条结束，微调好的模型会保存在`plant-dnagpt-BPE-promoter`目录中。在该目录中，会包含checkpoint目录，runs目录，以及保存好的微调模型。