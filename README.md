# Plant foundation DNA large language models (LLMs)

## 1. Environment

[Anaconda](https://docs.anaconda.com/free/anaconda/install/) package manager is recommended for building the training environment. For pre-train and fine-tune models, please ensure that you have a Nvidia GPU and the corresponding drivers are installed. For inference, devices without Nvidia GPU (CPU only, AMD GPU, Apple Silion, etc.) are also acceptable.

#### 1.1 Download and install [Anaconda](https://www.anaconda.com/download) package manager

#### 1.2 Create environment (We trained the models with python 3.11)

```bash
conda create -n llms python=3.11
conda activate llms
```

#### 1.3 Install dependencies

If you want to pre-train or fine-tune models, make sure you are using Nvidia GPU(s).  
Install [Nvidia driver](https://www.nvidia.com/download/index.aspx) and corresponding version of [CUDA driver](https://developer.nvidia.com/cuda-downloads) (> 11.0, we used CUDA 12.1).  

Also [Pytorch](https://pytorch.org/) (>=2.0) with corresponding CUDA version should also install.  
We recommend to use `pip` to install python packages that needed.  
```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

If you just want to use models for inference (prediction), you can install Pytorch GPU version (above) or install Pytorch CPU version if your machine has no Nvidia GPU.  
```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Next install other required dependencies.
```bash
git clone --recursive https://github.com/zhangtaolab/Plant_DNA_LLMs
cd Plant_DNA_LLMs
python3 -m pip install -r requirements.txt
```

(Optional) If you want to train a [mamba](https://github.com/state-spaces/mamba) model, you need to install several extra dependencies, also you should have a Nvidia GPU.
```bash
pip install causal-conv1d<=1.2.0
pip install mamba-ssm<2.0.0
```

### Pretrained Models
| Base model             | Model name                   | Params | Hugging Face | Model Scope |
| :--------------------: | :--------------------------: | :----: | :----------: | :---------: |
| BERT                   | Plant DNABERT                | 92M    | [download](https://huggingface.co/zhangtaolab/plant-dnabert) | [download](https://www.modelscope.cn/models/zhangtaolab/plant-dnabert) |
| GPT-2                  | Plant DNAGPT                 | 92M    | [download](https://huggingface.co/zhangtaolab/plant-dnagpt) | [download](https://www.modelscope.cn/models/zhangtaolab/plant-dnagpt) |
| Nucleotide Transformer | Plant Nucleotide Transformer | 102M   | [download](https://huggingface.co/zhangtaolab/plant-nucleotide-transformer) | [download](https://www.modelscope.cn/models/zhangtaolab/plant-nucleotide-transformer) |
| Gemma                  | Plant DNAGemma               | 152M   | [download](https://huggingface.co/zhangtaolab/plant-dnagemma) | [download](https://www.modelscope.cn/models/zhangtaolab/plant-dnagemma) |
| Mamba                  | Plant DNAMamba               | 130M   | [download](https://huggingface.co/zhangtaolab/plant-dnamamba) | [download](https://www.modelscope.cn/models/zhangtaolab/plant-dnamamba) |

## 2. Fine-tune

To fine-tune the plant DNA LLMs, please first download the desired models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local. You can use `git clone` (which may require `git-lfs` to be installed) to retrieve the model or directly download the model from the website.

In the activated `llms` python environment, use the `model_finetune.py` script to fine-tune a model for downstream task.  

Our script accepts `.csv` format data (separated by `,`) as input, when preparing the training data, please make sure the data contain a header and at least these two columns:
```csv
sequence,label
```
Where `sequence` is the input sequence, and `label` is the corresponding label for the sequence.

We also provide several plant genomic datasets for fine-tuning on the [HuggingFace](https://huggingface.co/zhangtaolab) and [ModelScope](https://www.modelscope.cn/organization/zhangtaolab).

With the appropriate supervised datasets, we can use the script to fine-tune a model for predicting promoters, for example:
```bash
python model_finetune.py \
    --model_name_or_path /path_to_the_model/plant-dnagpt \
    --train_data /path_to_the_data/train.csv \
    --test_data /path_to_the_data/test.csv \
    --eval_data /path_to_the_data/dev.csv \
    --train_task classification \
    --labels 'No;Yes' \
    --run_name plant_dnagpt_promoters \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --load_best_model_at_end \
    --metric_for_best_model 'f1' \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --output_dir finetune/plant-dnagpt-promoter
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

## 3. Inference

To use a fine-tuned model for inference, please first download the desired models from [HuggingFace](https://huggingface.co/zhangtaolab) or [ModelScope](https://www.modelscope.cn/organization/zhangtaolab) to local or provide a model trained by yourself.  

We also provide a script named `model_inference.py` for model inference.  
Here is an example that use the script to predict histone modification:
```bash
# Directly input a sequence
python model_inference.py -m /path_to_the_model/plant-dnagpt-H3K27ac -s sequence
# Provide a file contains multiple sequences to predict
python model_inference.py -m /path_to_the_model/plant-dnagpt-H3K27ac -f /path_to_the_data/data.txt -o results/H3K27ac.txt
```

In this script:
1. `-m`: Path to the fine-tuned model that is used for inference
2. `-s`: Input DNA sequence, only nucleotide A, C, G, T, N are acceptable
3. `-f`: Input file that contain multiple sequences, one line for each sequence. If you want to keep more information, file with `,` of `\t` separator is acceptable, but a header contains `sequence` column must be specified.

Output results contains the original sequence, input sequence length, predicted label and probability of each label (for regression task, will show a predicted score).


## Docker implementation for model inference

Environment deployment for LLMs may be an arduous job. To simplify this process, we also provide a docker version of our model inference code.

The images of the docker version are [here](https://hub.docker.com/r/zhangtaolab/plant_llms_inference), and the usage of docker implementation is shown below.  
For GPU inference (with Nvidia GPU), please pull the image with `gpu` tag, and make sure your computer has install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
```bash
docker pull zhangtaolab/plant_llms_inference:gpu
docker run --runtime=nvidia --gpus=all -v /Local_path:/Path_in_container zhangtaolab/plant_llms_inference:gpu -h
```
```bash
usage: inference.py [-h] [-v] -m MODEL [-f FILE] [-s SEQUENCE] [-t THRESHOLD]
                    [-l MAX_LENGTH] [-bs BATCH_SIZE] [-p SAMPLE] [-seed SEED]
                    [-d {cpu,gpu,mps,auto}] [-o OUTFILE] [-n]

Script for Plant DNA Large Language Models (LLMs) inference

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -m MODEL              Model path (should contain both model and tokenizer)
  -f FILE               File contains sequences that need to be classified
  -s SEQUENCE           One sequence that need to be classified
  -t THRESHOLD          Threshold for defining as True class (Default: 0.5)
  -l MAX_LENGTH         Max length of tokenized sequence (Default: 512)
  -bs BATCH_SIZE        Batch size for classification (Default: 1)
  -p SAMPLE             Subsampling for testing (Default: 1e7)
  -seed SEED            Random seed for subsampling (Default: None)
  -d {cpu,gpu,mps,auto}
                        Choose CPU or GPU to do inference (require specific
                        drivers) (Default: auto)
  -o OUTFILE            Prediction results (Default: stdout)
  -n                    Whether or not save the runtime locally (Default:
                        False)

Example:
  docker run --runtime=nvidia --gpus=all -v /local:/container zhangtaolab/plant_llms_inference:gpu -m model_path -f seqfile.csv -o output.txt
  docker run --runtime=nvidia --gpus=all -v /local:/container zhangtaolab/plant_llms_inference:gpu -m model_path -s 'ATCGGATCTCGACAGT' -o output.txt
```

For CPU inference,  please pull the image with `cpu` tag, this image support computer without NVIDIA GPU, such as cpu-only or Apple M-series Silicon. (Note that Inference of DNAMamba model is not supported in CPU mode)
```bash
docker pull zhangtaolab/plant_llms_inference:cpu
docker run -v /Local_path:/Path_in_container zhangtaolab/plant_llms_inference:cpu -h
```

The detailed usage is the same as the section [Inference](#3-inference).

### Demo for open chormtain prediction
we also provide demo server for open chormtain prediction by using Plant DNAMamba model.  
The web application is accessible at https://bioinfor.yzu.edu.cn/llms/open-chromatin/ or http://llms.zhangtaolab.org/llms/open-chromatin.

Preview:

![gradio](imgs/gradio.jpeg)

